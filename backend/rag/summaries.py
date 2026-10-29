# backend/rag/summaries.py
import os
import json
import re
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORAGE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "storage", "summaries"))
SUMMARIES_DIR = os.environ.get("SUMMARIES_DIR", DEFAULT_STORAGE)
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gemini-2.5-flash")
BUCKET_SEC = float(os.environ.get("SUMMARY_BUCKET_SEC", 120.0))  # 2 minutes per bucket


def _get_storage_dir() -> str:
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    return SUMMARIES_DIR


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_\.-]', '_', title)
    return f"{cleaned}.json"


def mmss(sec: float) -> str:
    s = int(max(0, sec))
    return f"{s // 60:02d}:{s % 60:02d}"


def format_summary_context(summary_data: Dict[str, Any]) -> str:
    """Formats structured summary data for prompt injection."""
    if not summary_data:
        return ""

    overview = summary_data.get("overview", "").strip()
    chapters = summary_data.get("chapters", [])

    parts = []
    if overview:
        parts.append(f"Global Video Overview:\n{overview}")

    if chapters:
        chapter_lines = ["\nVideo Chapters:"]
        for ch in chapters:
            start_fmt = mmss(ch.get("start", 0.0))
            end_fmt = mmss(ch.get("end", 0.0))
            title = ch.get("title", "Chapter").strip()
            summary = ch.get("summary", "").strip()
            chapter_lines.append(f"- [{start_fmt}–{end_fmt}] {title}: {summary}")
        parts.append("\n".join(chapter_lines))

    return "\n\n".join(parts)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts first valid JSON object from LLM response text."""
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # Try finding outermost braces
        m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


async def _call_llm(client: Any, prompt: str) -> str:
    if client is None:
        return ""
    try:
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            resp = await client.aio.models.generate_content(
                model=SUMMARY_MODEL,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 512},
            )
            return (resp.text or "").strip()
        elif hasattr(client, "models"):
            resp = client.models.generate_content(
                model=SUMMARY_MODEL,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 512},
            )
            return (resp.text or "").strip()
    except Exception as e:
        print(f"[summaries] LLM generation error: {e}")
    return ""


async def ensure_summary(client: Any, title: str) -> Dict[str, Any]:
    """
    Returns structured global video summary {overview: str, chapters: list}.
    Reads from persistent cache at storage/summaries/<title>.json if available.
    Otherwise, builds it hierarchically using Map-Reduce and caches it.
    """
    if not title:
        return {"overview": "No video specified.", "chapters": []}

    storage_dir = _get_storage_dir()
    cache_path = os.path.join(storage_dir, _safe_filename(title))

    # 1. Check persistent cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and data.get("overview"):
                return data
        except Exception as e:
            print(f"[summaries] Could not read cache {cache_path}: {e}")

    # 2. Query video transcript and frames from Pixeltable
    transcript_segments: List[Dict[str, Any]] = []
    frame_items: List[Dict[str, Any]] = []

    try:
        import pixeltable as pxt
        videos = pxt.get_table("video_rag.videos")
        v_rows = videos.where(videos.title == title).select(videos.transcript).collect()
        if v_rows and isinstance(v_rows[0].get("transcript"), dict):
            transcript_segments = v_rows[0]["transcript"].get("segments", [])

        frames = pxt.get_table("video_rag.frames")
        select_cols = [frames.pos, frames.combined_caption]
        if hasattr(frames, "timestamp_sec"):
            select_cols.append(frames.timestamp_sec)
        f_rows = (
            frames.where((frames.title == title) & (frames.combined_caption != None))
            .select(*select_cols)
            .order_by(frames.pos)
            .collect()
        )
        for r in f_rows:
            raw_t = r.get("timestamp_sec"); t = float(raw_t) if raw_t is not None else float(r["pos"] / FRAME_SAMPLE_FPS)
            frame_items.append({"time": t, "text": r["combined_caption"]})
    except Exception as e:
        print(f"[summaries] Database introspection error for {title}: {e}")

    if not transcript_segments and not frame_items:
        fallback = {"overview": f"No indexed content found for {title}.", "chapters": []}
        return fallback

    # Determine total duration
    max_t = 0.0
    if transcript_segments:
        max_t = max(max_t, max(s.get("end", 0.0) for s in transcript_segments))
    if frame_items:
        max_t = max(max_t, max(f["time"] for f in frame_items))
    max_t = max(max_t, 1.0)

    # 3. Create time buckets (e.g. 2 minutes each)
    buckets: List[Dict[str, Any]] = []
    cur_t = 0.0
    while cur_t < max_t:
        b_start = cur_t
        b_end = min(cur_t + BUCKET_SEC, max_t)

        b_segs = [
            s["text"].strip()
            for s in transcript_segments
            if s.get("start", 0.0) <= b_end and s.get("end", 0.0) >= b_start and s.get("text")
        ]
        # Sample frames every 5s within the bucket to prevent prompt bloat
        b_frames = [
            f["text"].strip()
            for f in frame_items
            if b_start <= f["time"] <= b_end and f.get("text") and int(f["time"]) % 5 == 0
        ]

        buckets.append({
            "start": round(b_start, 1),
            "end": round(b_end, 1),
            "speech": " ".join(b_segs),
            "visuals": " | ".join(b_frames[:10]),
        })
        cur_t += BUCKET_SEC

    # 4. Map Step: Summarize each bucket
    mapped_chapters: List[Dict[str, Any]] = []
    for i, b in enumerate(buckets, 1):
        content_desc = []
        if b["speech"]:
            content_desc.append(f"Spoken Speech: {b['speech'][:1200]}")
        if b["visuals"]:
            content_desc.append(f"Visual Descriptions: {b['visuals'][:1200]}")

        combined_desc = "\n".join(content_desc).strip()
        if not combined_desc:
            combined_desc = "Brief visual activity with no speech."

        map_prompt = f"""Summarize this segment of video '{title}' ({mmss(b['start'])} to {mmss(b['end'])}) in 2-3 sentences based strictly on the evidence below. Do not add external facts.
Evidence:
{combined_desc}

Return strictly a JSON object with:
{{"title": "Short 2-5 word topic title", "summary": "2-3 concise summary sentences"}}"""

        llm_out = await _call_llm(client, map_prompt)
        parsed = _extract_json(llm_out)
        if parsed and parsed.get("summary"):
            mapped_chapters.append({
                "start": b["start"],
                "end": b["end"],
                "title": parsed.get("title", f"Section {i}").strip(),
                "summary": parsed.get("summary", "").strip(),
            })
        else:
            # Deterministic rule-based fallback
            first_sentence = b["speech"].split(". ")[0] if b["speech"] else b["visuals"].split(". ")[0]
            first_sentence = first_sentence.strip()[:200]
            mapped_chapters.append({
                "start": b["start"],
                "end": b["end"],
                "title": f"Part {i} ({mmss(b['start'])}–{mmss(b['end'])})",
                "summary": first_sentence or f"Video activity between {mmss(b['start'])} and {mmss(b['end'])}.",
            })

    # 5. Reduce Step: Generate overarching overview
    if len(mapped_chapters) == 1:
        overview = mapped_chapters[0]["summary"]
    else:
        chapters_summary_text = "\n".join(
            f"[{mmss(c['start'])}–{mmss(c['end'])}] {c['title']}: {c['summary']}"
            for c in mapped_chapters
        )
        reduce_prompt = f"""Merge these chronological chapter summaries for video '{title}' into a cohesive high-level overview.
Chapters:
{chapters_summary_text}

Return strictly a JSON object with:
{{"overview": "3-5 sentence cohesive summary of the entire video"}}"""

        llm_out = await _call_llm(client, reduce_prompt)
        parsed = _extract_json(llm_out)
        if parsed and parsed.get("overview"):
            overview = parsed["overview"].strip()
        else:
            overview = " ".join(c["summary"] for c in mapped_chapters[:3])

    final_summary = {
        "title": title,
        "overview": overview,
        "chapters": mapped_chapters,
    }

    # 6. Save to cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        print(f"[summaries] Saved summary cache to {cache_path}")
    except Exception as e:
        print(f"[summaries] Could not write cache to {cache_path}: {e}")

    return final_summary
