"""backend/rag/clip_intent.py: Clip request intent detection and pre-roll calculation (FIX-15).

Provides:
- Regex-based intent detection for video clip requests
- Deterministic boundary calculation with pre-roll and minimum duration
"""

import re
from typing import Dict, Any, Tuple

WANT_CLIP = re.compile(
    r"\b(show|play|watch|clip|let me see|see the (?:part|moment|scene)|where (?:does|did))\b",
    re.IGNORECASE,
)


def want_clip(query: str) -> bool:
    """Returns True if the query explicitly asks to watch, see, or play a video clip."""
    if not query:
        return False
    return bool(WANT_CLIP.search(query.strip()))


def calculate_clip_boundaries(
    span: Dict[str, Any],
    pre_roll: float = 1.0,
    min_dur: float = 4.0,
    max_dur: float = 15.0,
    post_padding: float = 2.0,
) -> Tuple[float, float]:
    """Calculates deterministic start and duration for a clip from an evidence span.

    Applies pre-roll padding before span start to capture context, and bounded duration.
    """
    start_sec = float(span.get("start", span.get("timestamp_seconds", 0.0)))
    end_sec = float(span.get("end", start_sec + 1.0))

    clip_start = max(0.0, start_sec - pre_roll)
    raw_dur = (end_sec - start_sec) + post_padding
    clip_dur = min(max_dur, max(min_dur, raw_dur))

    return round(clip_start, 2), round(clip_dur, 2)
