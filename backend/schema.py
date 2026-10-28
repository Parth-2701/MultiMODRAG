#schema.py
import os
import sys


import subprocess as _subprocess
_original_subprocess_run = _subprocess.run
 
def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get('args', '')
    cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if 'pg_ctl' in cmd_str and kwargs.get('timeout') is not None:
        kwargs['timeout'] = max(kwargs['timeout'], 90)  # covers the ~35s AV lock delay with margin
    return _original_subprocess_run(*args, **kwargs)
 
_subprocess.run = _patched_subprocess_run
print("[schema] Patched subprocess.run: pg_ctl timeout extended to >=90s (Windows AV lock workaround)")

# Ensure 'schema' alias exists so Pixeltable catalog can resolve UDFs registered under either 'schema' or 'backend.schema'
sys.modules.setdefault("schema", sys.modules[__name__])

import pixeltable as pxt
from pixeltable.functions.gemini import embed_content
from pixeltable.functions.video import frame_iterator
from dotenv import load_dotenv

try:
    from backend.rag.embeddings import embed_query, embed_docs, DIM
    from backend.rag.index_version import check_or_update_index_version
except ImportError:
    from rag.embeddings import embed_query, embed_docs, DIM
    from rag.index_version import check_or_update_index_version

# 1. Load environment variables from .env file
load_dotenv()

# IMPORTANT: keep this in sync with FRAME_SAMPLE_FPS in main.py — both files
# independently need the same sampling rate to correctly convert frames.pos
# (a sequential frame index) into a real video timestamp in seconds.
FRAME_SAMPLE_FPS = 1.0

# 2. API key
#    GEMINI_API_KEY is the ONLY external key this file needs now. It's used for:
#      - captioning (gemini_caption below, primary path)
#      - embeddings (add_embedding_index below)
#      - and separately by main.py for chat response synthesis
#    Everything runs on Google AI Studio's free tier (no credit card, doesn't
#    expire) — you're just subject to per-minute/per-day rate limits rather than
#    the pay-per-request credits that HF Inference Providers uses.
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or api_key == "your-gemini-api-key-here":
    print("[warn]  GEMINI_API_KEY not set — needed for captioning, embeddings, "
          "and main.py's chat endpoint.")

# 3. Local fallback vision-language model: Florence-2 (microsoft/Florence-2-base)
#    ~0.23B params, ~460MB on disk, runs comfortably on CPU.
#    Loaded once at module import time so Pixeltable doesn't reload it per frame.
#    This only runs when the remote Gemini call fails (rate limit, network blip,
#    empty response, etc.), so captioning never hard-fails.
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"  # swap to "-large" (0.77B) for a quality bump

_florence_model = None
_florence_processor = None

def _get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        print(f"[schema] Loading Florence-2 ({FLORENCE_MODEL_ID}) on {DEVICE} as fallback captioner...")
        _florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
        ).to(DEVICE).eval()
        _florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID, trust_remote_code=True
        )
        print("[schema] Florence-2 fallback loaded.")
    return _florence_model, _florence_processor

FLORENCE_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def _run_florence(frame):
    """Raw Florence-2 inference on a single frame. Plain function (not a UDF) so it
    can be called both as its own UDF and as the fallback from inside
    gemini_caption without any cross-UDF-invocation weirdness.
    Returns None on failure (no dummy failure sentinels)."""
    if frame is None:
        return None
    try:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")

        model, processor = _get_florence()
        inputs = processor(
            text=FLORENCE_TASK_PROMPT, images=frame, return_tensors="pt"
        ).to(DEVICE, TORCH_DTYPE)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = processor.post_process_generation(
            generated_text,
            task=FLORENCE_TASK_PROMPT,
            image_size=(frame.width, frame.height),
        )

        caption = parsed.get(FLORENCE_TASK_PROMPT, "").strip()
        return caption if caption else None
    except Exception as e:
        print(f"[warn]  Florence-2 fallback also failed for one frame: {e}")
        return None


@pxt.udf
def florence_caption(frame: pxt.Image) -> str:
    """Standalone Florence-2 UDF, kept for direct use / debugging if you ever want
    to add it back as its own computed column."""
    return _run_florence(frame)


# --- Local audio transcription: faster-whisper (CTranslate2), not vanilla
# openai-whisper. ~4x faster on CPU and substantially lower memory with int8
# quantization — matters because this model is loaded into RAM alongside
# Florence-2, and free-tier deployment hosts (Render/Railway/HF Spaces)
# commonly cap memory at 512MB-1GB. Fully local: no API key, no rate limits,
# safe to deploy publicly (the "openai" in the pip package name just credits
# who released the original model weights — there is no call to
# api.openai.com anywhere in this path).
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")  # Multilingual base by default (FIX-12)
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print(f"[schema] Loading faster-whisper ({WHISPER_MODEL_SIZE})...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[schema] faster-whisper loaded.")
    return _whisper_model


def _run_transcribe_audio(audio) -> dict:
    """Core transcription logic decoupled from Pixeltable UDF wrapper for testability.

    Robust faster-whisper execution (FIX-12):
    - Multilingual model by default (WHISPER_MODEL_SIZE="base", or custom via env var)
    - Silero VAD filtering (min_silence_duration_ms=500) to strip non-speech
    - Hallucination gating (pruning no_speech_prob > 0.6 or avg_logprob < -1.0)
    - condition_on_previous_text=False to prevent infinite repetition loops
    - Explicit status contract: 'ok', 'silent', 'failed', or 'no_audio'
    """
    if audio is None:
        return {
            "status": "no_audio",
            "language": "",
            "language_probability": 0.0,
            "text": "",
            "segments": [],
        }
    try:
        whisper = _get_whisper()
        segments_iter, info = whisper.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        segs = []
        for s in segments_iter:
            # Drop segments flagged as non-speech or severe low-confidence hallucinations
            if getattr(s, "no_speech_prob", 0.0) > 0.6 or getattr(s, "avg_logprob", 0.0) < -1.0:
                continue
            clean_text = s.text.strip()
            if clean_text:
                segs.append({"start": float(s.start), "end": float(s.end), "text": clean_text})

        status = "ok" if segs else "silent"
        lang = getattr(info, "language", "en") if info else "en"
        lang_prob = getattr(info, "language_probability", 1.0) if info else 1.0

        return {
            "status": status,
            "language": lang,
            "language_probability": lang_prob,
            "text": " ".join(s["text"] for s in segs),
            "segments": segs,
        }
    except Exception as e:
        print(f"[warn]  faster-whisper transcription failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "language": "",
            "language_probability": 0.0,
            "text": "",
            "segments": [],
        }


@pxt.udf
def transcribe_audio(audio: pxt.Audio) -> dict:
    """Transcribes a video's extracted audio track using faster-whisper.
    Returns {'status': ..., 'language': ..., 'language_probability': ..., 'text': ..., 'segments': [...]}.
    Defensively distinguishes silent audio from failed transcription."""
    return _run_transcribe_audio(audio)


# --- Primary captioner: a cascade of Gemini models (all free tier). Free of
#     charge on Google AI Studio — no credit card, doesn't expire — but
#     rate-limited (RPM/RPD, not a spend-based credit pool like HF). Crucially,
#     rate limits are tracked PER MODEL, not shared across models, so trying
#     several models in sequence genuinely multiplies your effective free
#     throughput rather than just reordering the same quota.
#
#     Order below is best-quality-first, then models with progressively more
#     free-tier headroom, so you get the best caption whenever quota allows and
#     only degrade in quality when you have to — Florence-2 (local, unlimited)
#     is the very last resort after every Gemini option is exhausted.
from google import genai

GEMINI_CAPTION_MODELS = [
    # --- PRO TIER: Elite visual analysis, but tightest rate limits ---
    "gemini-3.1-pro-preview",           # Deepest reasoning, highest overall quality
    "gemini-3-pro-image",                # Dedicated Pro vision model, supreme pixel details
    "gemini-2.5-pro",                    # Stable generation Pro tier

    # --- FLASH TIER: High quality, standard production headroom ---
    "gemini-3.5-flash",                  # Best balance of new capability & speed
    "gemini-3.1-flash-image",            # State-of-the-art specialized flash vision
    "gemini-3-flash-preview",            # Strong 3-series fallback
    "gemini-2.5-flash",                  # Highly stable GA version
    "gemini-2.5-flash-image",            # 2.5-era specialized vision
    "gemini-2.0-flash",                  # Fast, classic 2.0 engine

    # --- LITE TIER: Maximum free-tier RPM/RPD headroom ---
    "gemini-3.1-flash-lite",             # Highest quality of the low-latency tiers
    "gemini-3.1-flash-lite-image",       # Specialized fast-vision endpoint
    "gemini-2.5-flash-lite",             # Most generous daily request quota (RPD)
    "gemini-2.0-flash-lite"              # Base baseline model for high-throughput safety net
]

from pydantic import BaseModel, Field
from typing import List

# FIX-10: Pydantic structured output model for honest frame descriptions & OCR
class FrameCaption(BaseModel):
    scene: str = ""                                     # environment + setting
    subjects: List[str] = Field(default_factory=list)  # people/characters/objects with visual attributes
    actions: str = ""                                   # directly observable physical actions
    on_screen_text: str = ""                            # verbatim transcript of slide titles, code, labels, subtitles
    shot: str = ""                                      # close-up, wide shot, slide, screen recording...


GEMINI_PROMPT = """You see ONE frame from a video. Describe only what is visible.
Fill the fields. Transcribe all readable on-screen text VERBATIM in on_screen_text (slide titles, code,
labels, numbers, subtitles); leave it empty if there is none. Do not infer identities, intentions,
dialogue, or events before/after this frame."""


def parse_and_assemble_caption(raw_text: str) -> str:
    """Parses JSON FrameCaption or falls back gracefully to raw string."""
    if not raw_text or not raw_text.strip():
        return ""
    text_to_parse = raw_text.strip()
    if text_to_parse.startswith("```"):
        text_to_parse = re.sub(r"^```(?:json)?\s*", "", text_to_parse, flags=re.IGNORECASE)
        text_to_parse = re.sub(r"\s*```$", "", text_to_parse)

    try:
        c = FrameCaption.model_validate_json(text_to_parse)
        subjects_str = " ".join(s.strip() for s in c.subjects if s.strip()) if isinstance(c.subjects, list) else str(c.subjects).strip()
        parts = [p for p in [c.scene.strip(), subjects_str, c.actions.strip()] if p]
        text = ". ".join(parts) if parts else (c.shot.strip() or "")
        if c.on_screen_text and c.on_screen_text.strip():
            text += f" Visible text: {c.on_screen_text.strip()}"
        return text.strip() or raw_text.strip()
    except Exception:
        return raw_text.strip()


import time
from typing import Optional, Dict, Any

_genai_client = genai.Client(api_key=api_key) if api_key else None

# FIX-11: Circuit breaker state per cascade model (until timestamp, consecutive failures)
_model_state: Dict[str, Dict[str, Any]] = {
    m: {"until": 0.0, "consecutive_fails": 0} for m in GEMINI_CAPTION_MODELS
}


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _is_model_usable(model_id: str) -> bool:
    """Checks whether the model's circuit-breaker cooldown has expired."""
    return time.time() >= _model_state.get(model_id, {}).get("until", 0.0)


def _record_success(model_id: str) -> None:
    """Resets consecutive failure counter on success."""
    if model_id in _model_state:
        _model_state[model_id]["consecutive_fails"] = 0


def _record_failure(model_id: str, exc: Exception) -> None:
    """Applies circuit-breaker penalties based on error category."""
    if model_id not in _model_state:
        _model_state[model_id] = {"until": 0.0, "consecutive_fails": 0}

    s = _model_state[model_id]
    err_str = str(exc)

    if "404" in err_str or "NOT_FOUND" in err_str:
        # Non-existent endpoint: drop permanently for this session
        s["until"] = float("inf")
        print(f"[circuit-breaker] Model '{model_id}' returned 404 (Not Found). Dropped permanently for this run.")
    elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        # Rate-limited: cooldown for 60 seconds
        s["until"] = time.time() + 60.0
        print(f"[circuit-breaker] Model '{model_id}' rate-limited (429). Cooled down for 60s.")
    else:
        s["consecutive_fails"] += 1
        if s["consecutive_fails"] >= 3:
            s["until"] = time.time() + 300.0
            s["consecutive_fails"] = 0
            print(f"[circuit-breaker] Model '{model_id}' failed 3 consecutive times. Cooled down for 5m.")


@pxt.udf
def gemini_caption(frame: pxt.Image) -> Optional[str]:
    """Generates a dense visual caption with dedicated OCR for a single video frame.
    Uses an active circuit breaker across the 12+ Gemini cascade models so exhausted
    or dead models are skipped with zero latency. Falls back to Florence-2 only if
    all Gemini options fail. Returns None on unindexable frames (no dummy sentinels)."""
    if frame is None:
        return None

    if _genai_client is None:
        return _run_florence(frame)

    if frame.mode != "RGB":
        frame = frame.convert("RGB")

    now = time.time()
    usable_models = [m for m in GEMINI_CAPTION_MODELS if _is_model_usable(m)]

    # If all models are in cooldown, briefly sleep up to min cooldown
    if not usable_models:
        min_until = min((_model_state[m]["until"] for m in GEMINI_CAPTION_MODELS), default=now)
        wait_sec = min(30.0, max(0.0, min_until - now))
        if wait_sec > 0:
            print(f"[circuit-breaker] All Gemini models cooling down. Waiting {wait_sec:.1f}s...")
            time.sleep(wait_sec)
        usable_models = [m for m in GEMINI_CAPTION_MODELS if _is_model_usable(m)]

    for model_id in usable_models:
        try:
            try:
                response = _genai_client.models.generate_content(
                    model=model_id,
                    contents=[frame, GEMINI_PROMPT],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": FrameCaption,
                        "temperature": 0.1,
                    },
                )
            except Exception:
                # Fallback to plain prompt call if model does not support structured schema
                response = _genai_client.models.generate_content(
                    model=model_id,
                    contents=[frame, GEMINI_PROMPT],
                )

            raw_caption = (response.text or "").strip()
            if raw_caption:
                assembled = parse_and_assemble_caption(raw_caption)
                if assembled:
                    _record_success(model_id)
                    return assembled
            print(f"[warn]  {model_id} returned an empty caption, trying next model...")
            _record_failure(model_id, RuntimeError("Empty caption returned"))
        except Exception as e:
            _record_failure(model_id, e)
            reason = "rate-limited" if _is_rate_limit_error(e) else f"{type(e).__name__}: {e!r}"
            print(f"[warn]  {model_id} failed ({reason}), trying next model in cascade...")
            continue

    print("[warn]  All Gemini cascade models exhausted for this frame, falling back to local Florence-2.")
    return _run_florence(frame)


# --- FIX-06: Text cleaning preserving grammatical phrasing, meaning, and casing ---
import re

_DISFLUENCY = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|h+m+)\b[.,?!;:]*\s*", re.I)
_WS = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    """Normalizes multiple whitespace characters into a single space and strips."""
    return _WS.sub(" ", text).strip() if text else ""


def _strip_fillers(text: str) -> str:
    """Removes vocal disfluencies (um, uh, erm, hm) from spoken audio while preserving
    grammatical words ('like', 'actually', 'you know', 'basically') and proper casing."""
    if not text or not text.strip():
        return ""
    cleaned = _DISFLUENCY.sub("", text)
    return _normalize_ws(cleaned)


def _compress_text(text: str) -> Optional[str]:
    """Normalizes whitespace in visual captions without lowercasing or destroying
    grammatical words. Returns None for empty frames (no failure sentinels)."""
    cleaned = _normalize_ws(text) if text else ""
    return cleaned if cleaned else None


def _clean_audio_text(text: str) -> str:
    """Strips vocal disfluencies from spoken audio text, returning empty string
    for silent frames (presence/absence of speech is meaningful)."""
    return _strip_fillers(text)


@pxt.udf
def compress_text(text: str) -> str:
    return _compress_text(text)


@pxt.udf
def clean_audio_text(text: str) -> str:
    return _clean_audio_text(text)


# --- Audio: align Whisper's segment-level transcript to each frame's real
# timestamp, clean it, and merge with the visual caption — all in ONE UDF.
#
# This used to be split across three chained computed columns
# (audio_at_timestamp -> clean_audio_text -> combine_captions), each
# depending on the previous. That chain silently produced null
# combined_caption values with ZERO error messages anywhere (confirmed via
# .errormsg on every column in the chain), even on a fully fresh DB reset —
# despite every function in the chain being written so it can never itself
# return None. That points to Pixeltable's incremental computation not
# correctly resolving a 3-level-deep dependency chain where one link
# (audio_caption_raw) combines a cross-table join (frames.transcript,
# inherited from the base videos table) with a view-native expression
# (frames.pos / FRAME_SAMPLE_FPS) in the same call. Collapsing everything
# into a single computed column removes that chain entirely.
# --- Audio: align Whisper's segment-level transcript to each frame's real
# timestamp, clean it, and merge with the visual caption.
#
# NOT a computed column, deliberately. Three different computed-column
# structures (a 3-column chain; a single collapsed UDF taking .segments via
# Pixeltable's expression syntax; a single collapsed UDF taking the whole
# transcript dict) all produced silent null results with ZERO errors
# anywhere, confirmed via direct inspection: frames.transcript reads back
# correctly as a real populated dict via plain select(), but the identical
# expression passed as a UDF argument inside add_computed_column never
# actually executes the UDF. This points to a genuine Pixeltable limitation
# — a view's computed column apparently can't reliably take another
# computed column from the joined BASE table as a UDF argument, even though
# that same column is perfectly readable via a normal query.
#
# Workaround: combined_caption is a plain column, filled in by ordinary
# Python (see backfill_combined_captions below) using data we've already
# proven reads correctly — sidesteps the computed-column DAG for this one
# cross-table dependency entirely.
def _compute_combined_caption_text(visual_caption: str, transcript: dict, timestamp: float) -> str:
    visual_caption = _normalize_ws(visual_caption)

    segments = []
    if isinstance(transcript, dict):
        segments = transcript.get('segments') or []

    window = 0.5 / FRAME_SAMPLE_FPS
    audio_text = ""
    if segments:
        matches = [
            seg['text'].strip()
            for seg in segments
            if seg['start'] <= timestamp + window and seg['end'] >= timestamp - window
        ]
        audio_text = _strip_fillers(" ".join(m for m in matches if m))

    if visual_caption and audio_text:
        return f"{visual_caption} Spoken audio: {audio_text}".strip()
    elif visual_caption:
        return visual_caption
    elif audio_text:
        return f"Spoken audio: {audio_text}".strip()
    return None


def backfill_combined_captions(title: str) -> None:
    """Fills in combined_caption for every frame of the given video, via
    plain Python + .update() calls — not a computed column. Call this once
    after a video's captioning has finished (i.e. after videos.insert()
    returns in main.py's process_video_background), before marking the
    task as completed."""
    videos = pxt.get_table('video_rag.videos')
    frames = pxt.get_table('video_rag.frames')

    video_rows = videos.where(videos.title == title).select(videos.transcript).collect()
    if len(video_rows) == 0:
        print(f"[warn]  backfill_combined_captions: no video row found for title={title!r}")
        return
    transcript = video_rows[0]['transcript']

    frame_rows = (
        frames.where(frames.title == title)
        .select(frames.pos, frames.clean_caption)
        .collect()
    )

    items = []
    texts_to_embed = []
    for row in frame_rows:
        timestamp = row['pos'] / FRAME_SAMPLE_FPS
        text = _compute_combined_caption_text(row['clean_caption'], transcript, timestamp)
        items.append((row['pos'], text))
        texts_to_embed.append(text or "")

    # Batched document embedding (RETRIEVAL_DOCUMENT, normalized, 768-dim) (FIX-13)
    vectors = []
    try:
        vectors = embed_docs(_genai_client, texts_to_embed, batch=100, dim=DIM)
    except Exception as e:
        print(f"[schema] Warning: embed_docs failed during backfill: {e}")
        vectors = [None] * len(items)

    for i, (pos, text) in enumerate(items):
        payload = {'combined_caption': text}
        if i < len(vectors) and vectors[i] is not None:
            payload['combined_emb'] = vectors[i]
        frames.where((frames.title == title) & (frames.pos == pos)).update(payload)

    print(f"[schema] Backfilled combined_caption and combined_emb for {len(frame_rows)} frame(s) of {title!r}.")


# 5. Wrap schema creation in a function to prevent transaction locks on import
def setup_db():
    print("Initializing Database Schema...")
    pxt.create_dir('video_rag', if_exists='ignore')

    # Define the primary Video Table
    videos = pxt.create_table(
        'video_rag.videos',
        {
            'video': pxt.Video,
            'title': pxt.String
        },
        if_exists='ignore'
    )

    # Audio extraction + local transcription (video-level, once per video —
    # NOT per frame, so this doesn't scale with frame count). extract_audio()
    # pulls the audio track from the uploaded file; transcribe_audio runs
    # faster-whisper on it. Added here, before the frames view is created,
    # so any newly-uploaded video has these ready before its frames are ever
    # computed (avoids a retroactive-backfill ordering issue).
    videos.add_computed_column(
        audio=videos.video.extract_audio(),
        if_exists='ignore'
    )
    videos.add_computed_column(
        transcript=transcribe_audio(videos.audio),
        if_exists='ignore'
    )

    # Create a Frame Extraction View (samples video at FRAME_SAMPLE_FPS).
    # NOTE: this directly drives how many Gemini requests you burn per video.
    # A 2-minute video at fps=1.0 = 120 frames = 120 requests. If you're on the
    # free tier and testing with several videos, consider lowering this to
    # leave headroom under the RPM/RPD caps.
    frames_view = pxt.create_view(
        'video_rag.frames',
        videos,
        iterator=frame_iterator(video=videos.video, fps=FRAME_SAMPLE_FPS),
        if_exists='ignore'
    )

    # Primary captioning pipeline: a cascade of Gemini models (free tier, see
    # GEMINI_CAPTION_MODELS above), with Florence-2 (local) as the final
    # automatic fallback — all baked into the UDF itself, so there's still
    # just one computed column here.
    frames_view.add_computed_column(
        caption_text=gemini_caption(frames_view.frame),
        if_exists='ignore'
    )

    # Apply Context Engineering compression to the caption text
    frames_view.add_computed_column(
        clean_caption=compress_text(frames_view.caption_text),
        if_exists='ignore'
    )

    # Combined visual + audio text — this is what actually gets embedded/searched.
    # PLAIN column (not computed) — see backfill_combined_captions above for
    # why. main.py must call schema.backfill_combined_captions(title) after
    # a video's captioning finishes, or this column stays null forever.
    frames_view.add_column(combined_caption=pxt.String, if_exists='ignore')

    # Dense Semantic Embedding Index — Gemini embeddings (remote, via Google's
    # Gemini API, also free tier), now over combined_caption (visual + audio)
    # instead of clean_caption alone, so retrieval covers both modalities.
    # FIX-14: Use if_exists='ignore' to stop destructive re-embedding on every server start
    frames_view.add_embedding_index(
        'combined_caption',
        idx_name='combined_caption_idx',
        embedding=embed_content.using(model='gemini-embedding-001'),
        if_exists='ignore'
    )

    # FIX-14: Track index configuration version to warn on drift instead of re-embedding
    check_or_update_index_version()

    # FIX-13: Parallel task-typed normalized embedding column & index
    frames_view.add_column(
        combined_emb=pxt.Array[(DIM,), pxt.Float],
        if_exists='ignore'
    )
    frames_view.add_embedding_index(
        'combined_emb',
        idx_name='combined_caption_emb_v2',
        string_embed=embed_query,
        metric='cosine',
        if_exists='ignore'
    )
    print("[schema] Schema setup complete.")


if __name__ == "__main__":
    setup_db()