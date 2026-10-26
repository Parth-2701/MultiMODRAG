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

import pixeltable as pxt
from pixeltable.functions.gemini import embed_content
from pixeltable.functions.video import frame_iterator
from dotenv import load_dotenv

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


def _run_florence(frame) -> str:
    """Raw Florence-2 inference on a single frame. Plain function (not a UDF) so it
    can be called both as its own UDF and as the fallback from inside
    gemini_caption without any cross-UDF-invocation weirdness."""
    if frame is None:
        return "unlabeled visual content"
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
        return caption if caption else "unlabeled visual content"
    except Exception as e:
        print(f"[warn]  Florence-2 fallback also failed for one frame: {e}")
        return "unlabeled visual content"


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

WHISPER_MODEL_SIZE = "base.en"  # drop to "tiny.en" if memory is tight on deploy
_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print(f"[schema] Loading faster-whisper ({WHISPER_MODEL_SIZE})...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[schema] faster-whisper loaded.")
    return _whisper_model


@pxt.udf
def transcribe_audio(audio: pxt.Audio) -> dict:
    """Transcribes a video's extracted audio track using faster-whisper.
    Returns {'text': ..., 'segments': [{'start', 'end', 'text'}, ...]}.
    Never raises: a failed/silent/missing audio track just yields empty
    segments, same defensive pattern as the Florence-2 fallback above — so a
    video with no audio (or audio faster-whisper can't parse) still gets
    fully indexed on its visual captions alone."""
    try:
        if audio is None:
            return {"text": "", "segments": []}
        whisper = _get_whisper()
        segments_iter, _info = whisper.transcribe(audio, beam_size=1)
        segments = []
        text_parts = []
        for seg in segments_iter:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
            text_parts.append(seg.text)
        return {"text": " ".join(text_parts).strip(), "segments": segments}
    except Exception as e:
        print(f"[warn]  faster-whisper transcription failed: {e}")
        return {"text": "", "segments": []}


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

GEMINI_PROMPT = """
You are generating descriptions for frames extracted from a continuous video.

Treat this image as one moment in an ongoing sequence rather than as an isolated photograph. The video may belong to any domain, including educational content, lectures, tutorials, movies, animation, sports, gameplay, documentaries, news, surveillance, vlogs, advertisements, or presentations.

Write a detailed factual description that captures everything useful for video retrieval and question answering.

Include:
- The overall scene or environment.
- Every significant person, character, animal, vehicle, or object.
- What each visible subject is doing.
- Interactions between subjects or objects.
- Relative positions (foreground/background, left/right/center).
- Any visible motion or changes occurring in this moment.
- Readable text, subtitles, labels, signs, or UI elements.
- Visual attributes such as colors, clothing, appearance, lighting, weather, and time of day if evident.
- Camera viewpoint if obvious (close-up, wide shot, aerial view, over-the-shoulder, etc.).

Describe only what is directly observable in this frame.
Do not infer identities, emotions, intentions, dialogue, or events that are not visually evident.
Do not speculate about what happened before or after this frame.
Write one coherent paragraph in natural language.
"""


_genai_client = genai.Client(api_key=api_key) if api_key else None


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


@pxt.udf
def gemini_caption(frame: pxt.Image) -> str:
    """Generates a dense visual caption for a single video frame by trying each
    model in GEMINI_CAPTION_MODELS in order (each has its own separate free-tier
    quota), falling back to local Florence-2 only if every Gemini model fails."""
    if frame is None:
        return "unlabeled visual content"

    if _genai_client is None:
        return _run_florence(frame)

    if frame.mode != "RGB":
        frame = frame.convert("RGB")

    for model_id in GEMINI_CAPTION_MODELS:
        try:
            response = _genai_client.models.generate_content(
                model=model_id,
                contents=[frame, GEMINI_PROMPT],
            )
            caption = (response.text or "").strip()
            if caption:
                return caption
            print(f"[warn]  {model_id} returned an empty caption, trying next model...")
        except Exception as e:
            reason = "rate-limited" if _is_rate_limit_error(e) else f"{type(e).__name__}: {e!r}"
            print(f"[warn]  {model_id} failed ({reason}), trying next model...")
            continue

    print("[warn]  All Gemini caption models failed for one frame, falling back to Florence-2.")
    return _run_florence(frame)


# --- Shared filler-word stripping, used by both visual and audio text cleaning ---
def _strip_fillers(text: str) -> str:
    if not text or not text.strip():
        return ""
    fillers = [' um ', ' uh ', ' you know ', ' like ', ' basically ', ' actually ']
    cleaned = " " + text.lower() + " "
    for filler in fillers:
        cleaned = cleaned.replace(filler, ' ')
    return cleaned.strip()


@pxt.udf
def compress_text(text: str) -> str:
    """Removes conversational filler words from the VISUAL caption. Keeps the
    'unlabeled visual content' fallback so downstream columns/embeddings never
    see an empty string for a frame that has no caption at all."""
    cleaned = _strip_fillers(text)
    return cleaned if cleaned else "unlabeled visual content"


@pxt.udf
def clean_audio_text(text: str) -> str:
    """Same filler-stripping as compress_text, but returns "" (not a fallback
    placeholder) for silent frames — presence/absence of speech is itself
    meaningful and shouldn't be papered over with fake text."""
    return _strip_fillers(text)


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
    visual_caption = (visual_caption or "").strip()

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

    if audio_text:
        return f"{visual_caption} Spoken audio: {audio_text}".strip()
    return visual_caption


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

    for row in frame_rows:
        timestamp = row['pos'] / FRAME_SAMPLE_FPS
        text = _compute_combined_caption_text(row['clean_caption'], transcript, timestamp)
        frames.where((frames.title == title) & (frames.pos == row['pos'])).update(
            {'combined_caption': text}
        )

    print(f"[schema] Backfilled combined_caption for {len(frame_rows)} frame(s) of {title!r}.")


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
    frames_view.add_embedding_index(
        'combined_caption',
        idx_name='combined_caption_idx',
        embedding=embed_content.using(model='gemini-embedding-001'),
        if_exists='replace'
    )
    print("[schema] Schema setup complete.")


if __name__ == "__main__":
    setup_db()