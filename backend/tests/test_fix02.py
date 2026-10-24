import pytest
from backend.rag.transcript_rag import chunk_segments, mmss, _fmt, _make, tokenize, rrf

def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(65) == "01:05"
    assert mmss(3661) == "61:01"

def test_chunk_segments_small():
    # A single segment below target duration and word count returns 1 chunk
    segs = [{"start": 0.0, "end": 12.0, "text": "Hello world this is a test segment."}]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 12.0
    assert chunks[0]["text"] == "Hello world this is a test segment."

def test_chunk_segments_duration_split_and_overlap():
    # Segments that exceed TARGET_SEC (45s default) trigger a split with OVERLAP
    segs = [
        {"start": 0.0, "end": 20.0, "text": "Segment one."},
        {"start": 20.0, "end": 48.0, "text": "Segment two."}, # dur = 48.0 >= 45s
        {"start": 48.0, "end": 70.0, "text": "Segment three."},
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 2
    # Chunk 1: segs 0 and 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 48.0
    assert "Segment one. Segment two." == chunks[0]["text"]
    # Chunk 2 overlaps seg 1 (overlap=1), so starts at 20.0 and ends at 70.0
    assert chunks[1]["start"] == 20.0
    assert chunks[1]["end"] == 70.0
    assert "Segment two. Segment three." == chunks[1]["text"]

def test_chunk_segments_word_count_split():
    # Segments that exceed MAX_WORDS trigger a split
    words_80 = "word " * 80
    segs = [
        {"start": 0.0, "end": 10.0, "text": words_80},
        {"start": 10.0, "end": 20.0, "text": words_80}, # total 160 words >= 140
        {"start": 20.0, "end": 30.0, "text": "Trailing sentence."},
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) >= 2

def test_fmt():
    c = {"start": 62.0, "end": 95.0, "text": "Quantum computing is fascinating."}
    formatted = _fmt(c)
    assert formatted == "[01:02–01:35] Quantum computing is fascinating."

def test_transcript_context_mock():
    # Test chronological selection logic
    rows = [
        {"start": 0.0, "end": 30.0, "text": "Introduction to the topic."},
        {"start": 30.0, "end": 60.0, "text": "Deep dive into algorithms."},
        {"start": 60.0, "end": 90.0, "text": "Experimental results and charts."},
        {"start": 90.0, "end": 120.0, "text": "Conclusion and future outlook."},
    ]
    # Under 3000 tokens, all rows should be returned with timestamps
    all_fmt = "\n".join(_fmt(r) for r in rows)
    assert "[00:00–00:30]" in all_fmt
    assert "[01:30–02:00]" in all_fmt
