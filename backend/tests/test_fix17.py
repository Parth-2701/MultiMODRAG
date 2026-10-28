"""Unit tests for FIX-17: Near-duplicate frame caption reuse (ingest optimization).

Verifies:
1. 64-bit dHash computation on grayscale gradients.
2. Sliding window near-duplicate caption reuse for identical / near-identical frames.
3. Sensitivity to visual changes (slide transitions, text edits) preventing false reuse.
4. Window eviction behavior (DEDUP_WINDOW).
5. gemini_caption fast-path invocation avoiding API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw
import numpy as np

import backend.schema as schema
from backend.schema import (
    _dhash,
    _get_reused_caption,
    _store_frame_caption,
    _recent_frames,
    gemini_caption,
)


@pytest.fixture(autouse=True)
def clear_dedup_cache():
    """Clear sliding window cache before and after each test."""
    _recent_frames.clear()
    yield
    _recent_frames.clear()


def test_dhash_computation():
    """dHash returns a 64-bit integer representing horizontal gradients."""
    img = Image.new("RGB", (64, 64), color="blue")
    h = _dhash(img)
    assert isinstance(h, int)
    assert h >= 0


def test_reused_caption_identical_frame():
    """Identical frame matches with 0 Hamming distance and 0.0 pixel diff."""
    img1 = Image.new("RGB", (128, 128), color="white")
    d1 = ImageDraw.Draw(img1)
    d1.rectangle((20, 20, 80, 80), fill="black")

    caption1 = "A high-contrast black square centered on a white background."
    _store_frame_caption(img1, caption1)

    # Identical copy
    img2 = img1.copy()
    reused = _get_reused_caption(img2)
    assert reused == caption1


def test_reused_caption_rejects_visual_change():
    """Altered image (different slide/scene) must not reuse caption."""
    img1 = Image.new("RGB", (128, 128), color="white")
    d1 = ImageDraw.Draw(img1)
    d1.text((10, 10), "Slide 1: Intro", fill="black")

    _store_frame_caption(img1, "Slide 1: Intro")

    # New slide with different graphics and text
    img2 = Image.new("RGB", (128, 128), color="white")
    d2 = ImageDraw.Draw(img2)
    d2.text((10, 10), "Slide 2: Architecture", fill="black")
    d2.ellipse((30, 30, 90, 90), fill="red")

    reused = _get_reused_caption(img2)
    assert reused is None


def test_sliding_window_expiration():
    """Sliding window evicts old frames when capacity (DEDUP_WINDOW) is exceeded."""
    # Insert DEDUP_WINDOW distinct frames
    window_size = _recent_frames.maxlen
    frames = []
    for i in range(window_size + 2):
        img = Image.new("RGB", (64, 64), color=(i * 30, (i * 40) % 256, (i * 50) % 256))
        frames.append(img)
        _store_frame_caption(img, f"Frame {i}")

    # First frame (index 0) must have been evicted from the window
    assert _get_reused_caption(frames[0]) is None
    # Most recent frame must still be cached
    assert _get_reused_caption(frames[-1]) == f"Frame {len(frames) - 1}"


def test_gemini_caption_reuses_and_skips_api():
    """gemini_caption immediately reuses cached caption for duplicate frames without API call."""
    img = Image.new("RGB", (64, 64), color="green")
    first_caption = "A green solid background frame."

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"summary": "A green solid background frame", "actions": [], "scene_type": "graphic", "detected_text": []}'
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(schema, "_genai_client", mock_client):
        fn = getattr(gemini_caption, "py_fn", gemini_caption)

        # 1st call: issues API request and stores caption
        cap1 = fn(img)
        assert "green solid background" in cap1
        assert mock_client.models.generate_content.call_count == 1

        # 2nd call with identical frame: should be served from cache without calling API
        cap2 = fn(img.copy())
        assert cap2 == cap1
        assert mock_client.models.generate_content.call_count == 1  # Unchanged!
