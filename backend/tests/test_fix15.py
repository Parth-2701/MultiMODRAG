"""Unit tests for FIX-15: Deterministic clip selection and correct tool-calling.

Verifies:
1. Clip intent detection regex (want_clip).
2. Deterministic boundary calculation with pre-roll padding and clamping.
3. FastMCP tool docstring accessibility.
4. Path traversal sanitization and fast-seek FFmpeg argument ordering.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from backend.rag.clip_intent import want_clip, calculate_clip_boundaries
import mcp_server.server as mcp_server


def test_want_clip_intent_detection():
    """want_clip detects explicit requests to watch, play, or show video clips."""
    assert want_clip("Show me where the rabbit jumps") is True
    assert want_clip("Can you play the clip of the speech?") is True
    assert want_clip("Let me see the part where the apple falls") is True
    assert want_clip("Where does the character enter the room?") is True
    assert want_clip("Watch this moment again") is True

    # Informational or summary queries should not trigger clip intent
    assert want_clip("Summarize the entire video") is False
    assert want_clip("What is the speaker's name?") is False
    assert want_clip("How many times did he repeat that word?") is False
    assert want_clip("") is False
    assert want_clip(None) is False


def test_calculate_clip_boundaries():
    """calculate_clip_boundaries adds pre-roll padding and enforces min/max duration bounds."""
    # Standard span: start=10.0, end=13.0 (3s span)
    span1 = {"start": 10.0, "end": 13.0}
    start1, dur1 = calculate_clip_boundaries(span1, pre_roll=1.0, min_dur=4.0, max_dur=15.0, post_padding=2.0)
    assert start1 == 9.0   # 10.0 - 1.0
    assert dur1 == 5.0     # (13.0 - 10.0) + 2.0 = 5.0

    # Near-zero start: pre-roll must not go below 0.0
    span2 = {"start": 0.5, "end": 2.0}
    start2, dur2 = calculate_clip_boundaries(span2, pre_roll=1.0, min_dur=4.0, max_dur=15.0)
    assert start2 == 0.0   # max(0.0, 0.5 - 1.0)
    assert dur2 == 4.0     # min_dur clamp

    # Long span: duration clamped to max_dur (15.0)
    span3 = {"start": 20.0, "end": 50.0}
    start3, dur3 = calculate_clip_boundaries(span3, pre_roll=1.0, min_dur=4.0, max_dur=15.0)
    assert start3 == 19.0
    assert dur3 == 15.0    # max_dur clamp


def test_mcp_extract_video_clip_docstring():
    """extract_video_clip.__doc__ must not be None so FastMCP exposes it to LLM schemas."""
    doc = mcp_server.extract_video_clip.__doc__
    assert doc is not None
    assert len(doc.strip()) > 20
    assert "Cuts a specific segment from a stored video file" in doc
    assert "video_filename" in doc
    assert "start_time" in doc


def test_mcp_extract_video_clip_sanitization_and_command():
    """extract_video_clip sanitizes filename traversal and uses -ss BEFORE -i in FFmpeg."""
    # 1. Non-existent file returns clean error
    res_missing = mcp_server.extract_video_clip("does_not_exist_xyz.mp4", start_time=5.0)
    assert "was not found in the videos directory" in res_missing

    # 2. Directory traversal attempt is stripped by os.path.basename
    res_traversal = mcp_server.extract_video_clip("../../etc/passwd", start_time=5.0)
    assert "passwd" in res_traversal
    assert ".." not in res_traversal

    # 3. Verify FFmpeg command construction with fast keyframe seeking
    mock_run = MagicMock()
    with patch("os.path.exists", return_value=True):
        with patch("subprocess.run", mock_run):
            clip_res = mcp_server.extract_video_clip("test_video.mp4", start_time=12.34, duration=6.5)
            assert clip_res.startswith("clips/clip_")

            cmd = mock_run.call_args[0][0]
            # Must place -ss BEFORE -i
            ss_idx = cmd.index("-ss")
            i_idx = cmd.index("-i")
            assert ss_idx < i_idx
            assert cmd[ss_idx + 1] == "12.34"

            # Must contain faststart for streaming playback
            assert "-movflags" in cmd
            assert "+faststart" in cmd
