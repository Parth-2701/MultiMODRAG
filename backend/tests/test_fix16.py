"""Unit tests for FIX-16: Store real timestamps at ingest.

Verifies:
1. Schema registers timestamp_sec as a computed column on frames_view.
2. hybrid_search reads stored timestamp_sec rather than dynamically computing pos / fps.
3. Changing runtime FPS parameter does not corrupt or drift timestamps for existing frames.
4. Backward compatibility: falls back to pos / fps when timestamp_sec is absent on legacy rows.
"""

import inspect
from unittest.mock import MagicMock, patch
import pytest

import backend.schema as schema
import backend.rag.hybrid_retrieval as hr


def test_schema_has_timestamp_sec_computed_column():
    """Verify that schema.setup_db defines timestamp_sec on frames_view."""
    src = inspect.getsource(schema.setup_db)
    assert "timestamp_sec=frames_view.pos / float(FRAME_SAMPLE_FPS)" in src
    assert "add_computed_column" in src


def test_hybrid_search_prefers_stored_timestamp_sec_over_runtime_fps():
    """Verify that hybrid_search uses stored timestamp_sec, immune to runtime fps changes.

    If a video was ingested at 1.0 fps, frame pos=10 has timestamp_sec=10.0.
    Even if the query passes fps=0.5 (which would yield 10/0.5 = 20.0),
    hybrid_search must report the true stored 10.0 seconds.
    """
    mock_frames = MagicMock()
    mock_frames.combined_caption = MagicMock()
    mock_frames.timestamp_sec = MagicMock()

    # Create dummy dense row with pos=10, timestamp_sec=10.0
    dummy_row = {
        "pos": 10,
        "timestamp_sec": 10.0,
        "combined_caption": "A bunny hopping across a grassy lawn",
        "video": "storage/videos/bunny.mp4",
        "title": "bunny.mp4",
    }

    with patch("backend.rag.hybrid_retrieval._lexical_index", return_value=(None, None, [dummy_row])):
        with patch.object(mock_frames, "where") as mock_where:
            mock_query = MagicMock()
            mock_query.order_by.return_value.limit.return_value.select.return_value.collect.return_value = [dummy_row]
            mock_where.return_value = mock_query

            # Pass runtime fps=0.5: if unpatched, 10 / 0.5 = 20.0
            hits = hr.hybrid_search(mock_frames, "bunny.mp4", "bunny hopping", fps=0.5)

            assert len(hits) == 1
            # Must match the stored timestamp_sec (10.0), NOT 20.0
            assert hits[0]["timestamp_seconds"] == 10.0
            assert hits[0]["pos"] == 10


def test_hybrid_search_fallback_when_timestamp_sec_missing():
    """Legacy rows without timestamp_sec safely fall back to pos / fps."""
    mock_frames = MagicMock()
    mock_frames.combined_caption = MagicMock()
    # Deliberately do not set timestamp_sec attribute on mock_frames
    del mock_frames.timestamp_sec

    dummy_legacy_row = {
        "pos": 30,
        "combined_caption": "Old legacy indexed frame",
        "video": "storage/videos/old.mp4",
        "title": "old.mp4",
    }

    with patch("backend.rag.hybrid_retrieval._lexical_index", return_value=(None, None, [dummy_legacy_row])):
        with patch.object(mock_frames, "where") as mock_where:
            mock_query = MagicMock()
            mock_query.order_by.return_value.limit.return_value.select.return_value.collect.return_value = [dummy_legacy_row]
            mock_where.return_value = mock_query

            hits = hr.hybrid_search(mock_frames, "old.mp4", "legacy", fps=1.0)
            assert len(hits) == 1
            assert hits[0]["timestamp_seconds"] == 30.0  # 30 / 1.0
