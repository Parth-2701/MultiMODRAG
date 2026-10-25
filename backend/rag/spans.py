# backend/rag/spans.py
import os
import re
from typing import List, Dict, Any, Tuple

GAP_SEC = float(os.environ.get("SPAN_GAP_SEC", 2.5))     # frames closer than this join one span
MAX_SPAN = float(os.environ.get("SPAN_MAX_SEC", 20.0))   # max duration of one span in seconds
MAX_SPANS = int(os.environ.get("MAX_SPANS", 6))          # max spans returned to prompt
POOL_SIZE = int(os.environ.get("SPAN_POOL_SIZE", 30))    # pool size of hits to cluster

SPEECH_SPLIT = re.compile(r"\s*Spoken audio:\s*", re.I)

def split_caption(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    parts = SPEECH_SPLIT.split(text, maxsplit=1)
    visual = parts[0].strip() if parts else ""
    speech = parts[1].strip() if len(parts) > 1 else ""
    return visual, speech

def _near_dup(v: str, existing: List[str], threshold: float = 0.8) -> bool:
    a = set(v.lower().split())
    if not a:
        return True
    for e in existing:
        b = set(e.lower().split())
        if not b:
            continue
        jaccard = len(a & b) / max(1, len(a | b))
        if jaccard >= threshold:
            return True
    return False

def to_spans(hits: List[Dict[str, Any]], fps: float = 1.0) -> List[Dict[str, Any]]:
    """
    Groups retrieved frame hits into cohesive, deduplicated temporal spans.
    hits: list of hit dicts containing 'timestamp_seconds', 'caption', 'score', 'source_filename'
    """
    if not hits:
        return []

    # Take generous pool of top-ranked hits before merging
    pool_hits = hits[:POOL_SIZE]
    # Sort pool chronologically for clustering
    pool = sorted(pool_hits, key=lambda h: h["timestamp_seconds"])

    groups: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []

    for h in pool:
        if cur and (
            (h["timestamp_seconds"] - cur[-1]["timestamp_seconds"] > GAP_SEC)
            or (h["timestamp_seconds"] - cur[0]["timestamp_seconds"] >= MAX_SPAN)
        ):
            groups.append(cur)
            cur = []
        cur.append(h)
    if cur:
        groups.append(cur)

    spans: List[Dict[str, Any]] = []
    frame_dur = 1.0 / max(0.001, fps)

    for g in groups:
        visuals: List[str] = []
        speeches: List[str] = []
        seen_speech: set[str] = set()

        for h in g:
            v, s = split_caption(h.get("caption", ""))
            if v and not _near_dup(v, visuals):
                visuals.append(v)
            if s:
                s_norm = s.strip()
                if s_norm and s_norm.lower() not in seen_speech:
                    speeches.append(s_norm)
                    seen_speech.add(s_norm.lower())

        start_t = g[0]["timestamp_seconds"]
        end_t = g[-1]["timestamp_seconds"] + frame_dur
        best_score = max(h.get("score", 0.0) for h in g)

        text_parts = []
        if visuals:
            text_parts.append(" ".join(visuals))
        if speeches:
            text_parts.append(f"Spoken: {' '.join(speeches)}")

        combined_text = " ".join(text_parts).strip()
        if not combined_text and g:
            combined_text = g[0].get("caption", "")

        spans.append({
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "text": combined_text,
            "score": best_score,
            "source_filename": g[0].get("source_filename", "video.mp4"),
            "frame_count": len(g),
        })

    # Rank spans by best supporting score
    spans.sort(key=lambda s: -s["score"])
    return spans[:MAX_SPANS]
