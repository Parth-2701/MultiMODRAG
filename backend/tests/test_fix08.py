"""
Unit tests for FIX-08: Query routing and stored global summary.
Verifies query classification, policy selection, summary formatting, and caching.
"""
import pytest
import os
import json
from unittest.mock import AsyncMock, MagicMock
from backend.rag.routing import route, get_policy, CLASSES
from backend.rag.summaries import (
    ensure_summary,
    format_summary_context,
    _extract_json,
    _safe_filename,
)
from backend.rag.prompting import build_user_content


def test_routing_classification_rules():
    # Summary queries
    assert route("Summarize the entire video for me") == "summary"
    assert route("Give me a brief overview of what this video is about") == "summary"
    assert route("What are the main points of the presentation?") == "summary"
    assert route("tl;dr of this clip") == "summary"

    # Temporal queries
    assert route("What happened after the apple fell?") == "temporal"
    assert route("What did he say before untying his top knot?") == "temporal"
    assert route("Show me the sequence of events") == "temporal"
    assert route("What was the next planet named?") == "temporal"

    # Aggregate queries
    assert route("How many times did the rabbit appear?") == "aggregate"
    assert route("Count the number of times Dhananand is addressed") == "aggregate"
    assert route("How often does the butterfly flutter?") == "aggregate"

    # Clip requests
    assert route("Show me the clip where the apple drops") == "clip_request"
    assert route("Can I watch the moment he makes the vow?") == "clip_request"
    assert route("Play the scene with the purple butterfly") == "clip_request"

    # Pinpoint / needle queries (fail-open default)
    assert route("What learning rate did the speaker use?") == "needle"
    assert route("In whose court does the video take place?") == "needle"
    assert route("What color is the rabbit?") == "needle"
    assert route("") == "needle"
    assert route("   ") == "needle"


def test_routing_policies():
    for c in CLASSES:
        policy = get_policy(c)
        assert "k_spans" in policy
        assert "use_summary" in policy
        assert "order" in policy
        assert "warn_approximate" in policy

    assert get_policy("summary")["k_spans"] == 4
    assert get_policy("summary")["use_summary"] is True

    assert get_policy("temporal")["k_spans"] == 14
    assert get_policy("temporal")["order"] == "chronological"

    assert get_policy("aggregate")["k_spans"] == 20
    assert get_policy("aggregate")["warn_approximate"] is True

    assert get_policy("needle")["k_spans"] == 6
    assert get_policy("needle")["use_summary"] is False


def test_extract_json():
    assert _extract_json('{"overview": "Good video", "chapters": []}') == {
        "overview": "Good video",
        "chapters": [],
    }

    # Markdown fenced
    fenced = '```json\n{"overview": "Fenced video", "chapters": []}\n```'
    assert _extract_json(fenced) == {"overview": "Fenced video", "chapters": []}

    # Malformed or empty
    assert _extract_json("") is None
    assert _extract_json("not json") is None


def test_format_summary_context():
    data = {
        "overview": "An overview of ancient Indian court history.",
        "chapters": [
            {"start": 0.0, "end": 15.0, "title": "Assembly", "summary": "Chanakya warns the king."},
            {"start": 15.0, "end": 37.0, "title": "The Vow", "summary": "Chanakya unties his shika."},
        ],
    }
    formatted = format_summary_context(data)
    assert "Global Video Overview:" in formatted
    assert "An overview of ancient Indian court history." in formatted
    assert "[00:00–00:15] Assembly: Chanakya warns the king." in formatted
    assert "[00:15–00:37] The Vow: Chanakya unties his shika." in formatted


@pytest.mark.anyio
async def test_ensure_summary_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.rag.summaries.SUMMARIES_DIR", str(tmp_path))

    title = "test_video.mp4"
    cache_file = tmp_path / _safe_filename(title)
    cached_content = {
        "title": title,
        "overview": "Cached summary overview.",
        "chapters": [{"start": 0.0, "end": 10.0, "title": "Ch1", "summary": "Intro"}],
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_content, f)

    # ensure_summary should read directly from disk cache without calling any client
    res = await ensure_summary(None, title)
    assert res["overview"] == "Cached summary overview."
    assert len(res["chapters"]) == 1


def test_build_user_content_summary_and_approximate_notes():
    question = "How many times does the butterfly fly?"
    visuals = [(0.0, "Rabbit resting"), (5.0, "Butterfly flying")]
    speech = "[00:00-00:05] Listen to the sounds"

    content = build_user_content(
        question=question,
        visual_items=visuals,
        speech_ctx=speech,
        video_name="sample.mp4",
        summary_ctx="Overview: A rabbit and a butterfly in a meadow.",
        warn_approximate=True,
    )

    assert "<summary_evidence>" in content
    assert "Overview: A rabbit and a butterfly in a meadow." in content
    assert "</summary_evidence>" in content
    assert "Video frames are sampled once per second; counts and frequencies are approximate" in content
