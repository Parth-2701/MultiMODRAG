import pytest
from backend.rag.spans import to_spans, split_caption, _near_dup, GAP_SEC, MAX_SPAN, MAX_SPANS

def test_split_caption():
    text = "A rabbit in green meadow. Spoken audio: Hello everyone welcome to the show."
    v, s = split_caption(text)
    assert v == "A rabbit in green meadow."
    assert s == "Hello everyone welcome to the show."

    # No spoken audio
    v2, s2 = split_caption("Only visual description here.")
    assert v2 == "Only visual description here."
    assert s2 == ""

def test_near_dup_jaccard():
    t1 = "A large white rabbit stands in a sunny green meadow"
    t2 = "A large white rabbit standing in a sunny green meadow"
    assert _near_dup(t2, [t1]) is True

    t3 = "A red car driving on a highway"
    assert _near_dup(t3, [t1]) is False

def test_speech_deduplication_in_span():
    # Consecutive frames with identical spoken text
    fps = 1.0
    hits = [
        {"timestamp_seconds": 10.0, "caption": "Scene A. Spoken audio: Welcome to our lecture.", "score": 0.5, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 11.0, "caption": "Scene A with rabbit. Spoken audio: Welcome to our lecture.", "score": 0.6, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 12.0, "caption": "Scene A close up. Spoken audio: Welcome to our lecture.", "score": 0.7, "source_filename": "vid.mp4"},
    ]
    spans = to_spans(hits, fps=fps)
    assert len(spans) == 1
    span = spans[0]
    assert span["start"] == 10.0
    assert span["end"] == 13.0
    # Spoken sentence must appear exactly ONCE
    assert span["text"].count("Welcome to our lecture.") == 1

def test_distinct_time_regions_coverage():
    # 3 distinct clusters: 0-2s, 40-42s, 100-102s
    fps = 1.0
    hits = [
        {"timestamp_seconds": 0.0, "caption": "Intro frame 0", "score": 0.9, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 1.0, "caption": "Intro frame 1", "score": 0.85, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 40.0, "caption": "Middle frame 40", "score": 0.8, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 41.0, "caption": "Middle frame 41", "score": 0.75, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 100.0, "caption": "Outro frame 100", "score": 0.7, "source_filename": "vid.mp4"},
        {"timestamp_seconds": 101.0, "caption": "Outro frame 101", "score": 0.65, "source_filename": "vid.mp4"},
    ]
    spans = to_spans(hits, fps=fps)
    assert len(spans) == 3
    starts = [s["start"] for s in spans]
    assert 0.0 in starts
    assert 40.0 in starts
    assert 100.0 in starts

def test_max_span_duration_constraint():
    # 35 continuous frames at 1 fps
    fps = 1.0
    hits = [
        {"timestamp_seconds": float(i), "caption": f"Frame {i}", "score": 0.1 * i, "source_filename": "vid.mp4"}
        for i in range(35)
    ]
    spans = to_spans(hits, fps=fps)
    max_allowed = MAX_SPAN + (1.0 / fps)
    for s in spans:
        duration = s["end"] - s["start"]
        assert duration <= max_allowed + 1e-4

def test_empty_hits():
    assert to_spans([], fps=1.0) == []
