"""backend/rag/visual_index.py: CLIP multimodal visual index and image queries (FIX-18).

Provides:
- Base64 image decoding for visual queries
- Zero-shot text-to-image and image-to-image visual retrieval over raw video frames
- Fail-open fallback when visual index is unavailable
"""

import base64
import io
import re
from typing import Optional, List, Any
from PIL import Image

_B64_HEADER_RE = re.compile(r"^data:image/[a-zA-Z0-9.+_-]+;base64,", re.IGNORECASE)


def decode_image_base64(b64_str: Optional[str]) -> Optional[Image.Image]:
    """Decodes a base64 string (with or without data URI header) into an RGB PIL Image."""
    if not b64_str or not b64_str.strip():
        return None

    cleaned = _B64_HEADER_RE.sub("", b64_str.strip())
    try:
        raw_bytes = base64.b64decode(cleaned)
        img = Image.open(io.BytesIO(raw_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"[visual_index] Warning: Failed to decode base64 image: {e}")
        return None


def visual_rank(
    frames: Any,
    title: Optional[str] = None,
    query_text: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    k: int = 30,
) -> List[int]:
    """Retrieves ranked frame positions using the direct CLIP visual embedding index.

    Supports:
    - Text-to-image: frames.frame.similarity(string=query_text)
    - Image-to-image: frames.frame.similarity(image=query_image)

    Returns:
        List of integer frame positions (pos) ordered by descending visual relevance.
        Fails open and returns [] if the visual index is absent or errors out.
    """
    if query_image is None and (not query_text or not query_text.strip()):
        return []

    try:
        if not hasattr(frames, "frame"):
            return []

        flt = (frames.title == title) if title is not None else None

        if query_image is not None:
            sim = frames.frame.similarity(image=query_image)
        else:
            sim = frames.frame.similarity(string=query_text.strip())

        query = frames
        if flt is not None:
            query = query.where(flt)

        rows = query.order_by(sim, asc=False).limit(k).select(frames.pos).collect()
        return [int(r["pos"]) for r in rows]
    except Exception as e:
        # Fails open: if visual index is not created or CLIP fails, search continues
        print(f"[visual_index] Notice: Visual ranking skipped/unavailable: {e}")
        return []
