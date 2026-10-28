"""Unit tests for FIX-18: CLIP visual index and multimodal image queries.

Verifies:
1. Base64 image decoding (with and without data URI scheme, RGBA conversion).
2. Zero-shot visual ranking via text queries: frames.frame.similarity(string=...).
3. Image-to-image visual ranking via image queries: frames.frame.similarity(image=...).
4. Tri-modal RRF fusion (dense + sparse + visual) in hybrid_search.
5. Fail-open resilience when visual index is absent or errors out.
"""

import io
import base64
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from backend.rag.visual_index import (
    decode_image_base64,
    visual_rank,
)
import backend.rag.hybrid_retrieval as hr


def _create_sample_base64_image(color="red", size=(32, 32)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_image_base64_valid():
    """decode_image_base64 decodes raw base64 and data URI schemes into RGB Images."""
    raw_b64 = _create_sample_base64_image(color="green", size=(40, 50))
    img = decode_image_base64(raw_b64)
    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.size == (40, 50)
    assert img.mode == "RGB"

    # With Data URI header
    uri_b64 = f"data:image/png;base64,{raw_b64}"
    img2 = decode_image_base64(uri_b64)
    assert img2 is not None
    assert img2.size == (40, 50)


def test_decode_image_base64_invalid_inputs():
    """Invalid or empty base64 strings safely return None."""
    assert decode_image_base64(None) is None
    assert decode_image_base64("") is None
    assert decode_image_base64("   ") is None
    assert decode_image_base64("corrupted!not!valid!base64!%%%") is None


def test_visual_rank_text_query():
    """visual_rank calls frames.frame.similarity(string=...) for text queries."""
    mock_frames = MagicMock()
    mock_frames.frame = MagicMock()
    mock_frames.title = "sample.mp4"

    mock_sim = MagicMock()
    mock_frames.frame.similarity.return_value = mock_sim

    mock_query = MagicMock()
    mock_query.order_by.return_value.limit.return_value.select.return_value.collect.return_value = [
        {"pos": 5}, {"pos": 12}, {"pos": 28}
    ]
    mock_frames.where.return_value = mock_query

    ranked = visual_rank(mock_frames, title="sample.mp4", query_text="a red sports car", k=10)

    assert ranked == [5, 12, 28]
    mock_frames.frame.similarity.assert_called_once_with(string="a red sports car")


def test_visual_rank_image_query():
    """visual_rank calls frames.frame.similarity(image=...) for image queries."""
    mock_frames = MagicMock()
    mock_frames.frame = MagicMock()

    mock_sim = MagicMock()
    mock_frames.frame.similarity.return_value = mock_sim

    mock_query = MagicMock()
    mock_query.order_by.return_value.limit.return_value.select.return_value.collect.return_value = [
        {"pos": 3}, {"pos": 9}
    ]
    mock_frames.where.return_value = mock_query

    query_img = Image.new("RGB", (64, 64), color="yellow")
    ranked = visual_rank(mock_frames, title="sample.mp4", query_image=query_img, k=10)

    assert ranked == [3, 9]
    mock_frames.frame.similarity.assert_called_once_with(image=query_img)


def test_visual_rank_fail_open_on_exception():
    """visual_rank returns [] when visual index errors out or is missing."""
    mock_frames = MagicMock()
    mock_frames.frame.similarity.side_effect = RuntimeError("Index 'frame_clip_idx' does not exist")

    ranked = visual_rank(mock_frames, title="sample.mp4", query_text="whiteboard", k=10)
    assert ranked == []


def test_hybrid_search_tri_modal_fusion():
    """hybrid_search fuses dense, sparse, and visual lists when visual_rank returns hits."""
    mock_frames = MagicMock()
    mock_frames.combined_caption = MagicMock()

    # Create mock rows for frames
    rows = [
        {"pos": 1, "combined_caption": "Frame 1 text", "video": "v.mp4", "title": "v.mp4", "timestamp_sec": 1.0},
        {"pos": 2, "combined_caption": "Frame 2 text", "video": "v.mp4", "title": "v.mp4", "timestamp_sec": 2.0},
        {"pos": 3, "combined_caption": "Frame 3 text", "video": "v.mp4", "title": "v.mp4", "timestamp_sec": 3.0},
    ]

    # Dense returns pos 1, 2
    mock_dense = [{"pos": 1, "combined_caption": "Frame 1 text", "video": "v.mp4", "title": "v.mp4", "timestamp_sec": 1.0}]
    mock_query = MagicMock()
    mock_query.order_by.return_value.limit.return_value.select.return_value.collect.return_value = mock_dense
    mock_frames.where.return_value = mock_query

    # Sparse (BM25) returns pos 2
    with patch("backend.rag.hybrid_retrieval._lexical_index", return_value=(None, MagicMock(get_scores=lambda q: [0, 1.5, 0]), rows)):
        # Visual rank returns pos 3
        with patch("backend.rag.visual_index.visual_rank", return_value=[3]):
            hits = hr.hybrid_search(mock_frames, "v.mp4", "query", fps=1.0)
            hit_positions = [h["pos"] for h in hits]

            # All three modalities contributed to the fusion
            assert 1 in hit_positions
            assert 2 in hit_positions
            assert 3 in hit_positions
