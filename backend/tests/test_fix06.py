"""
Unit tests for FIX-06: Stop destroying text before embedding.
Verifies disfluency stripping, casing preservation, and whitespace normalization.
"""
import pytest
from backend.schema import (
    _strip_fillers,
    _normalize_ws,
    _compress_text,
    _clean_audio_text,
    compress_text,
    clean_audio_text,
    _compute_combined_caption_text,
)


def test_acceptance_strip_fillers_and_compress_text():
    # 1. Acceptance test: speech filler stripping
    res_speech = _strip_fillers("Um, I like it, you know")
    assert res_speech == "I like it, you know"

    # 2. Acceptance test: caption compression preserves words and casing
    res_caption = _compress_text("A Dog looks like a Wolf")
    assert res_caption == "A Dog looks like a Wolf"
    # Pixeltable py_fn property also gives identical result
    assert compress_text.py_fn("A Dog looks like a Wolf") == "A Dog looks like a Wolf"


def test_strip_fillers_disfluencies():
    # True disfluencies (um, uh, erm, hm) stripped cleanly
    assert _strip_fillers("uh... well, um, actually basically") == "well, actually basically"
    assert _strip_fillers("Hmm, let me see, erm, that is right.") == "let me see, that is right."
    assert _strip_fillers("uh uh um") == ""


def test_strip_fillers_preserves_grammatical_words_and_casing():
    # Words like 'like', 'actually', 'basically', 'you know' are preserved
    text = "I actually like this framework basically because you know it scales."
    assert _strip_fillers(text) == text

    # Casing and proper nouns must survive
    cased = "Alexander the Great, um, founded Alexandria in Egypt."
    assert _strip_fillers(cased) == "Alexander the Great, founded Alexandria in Egypt."


def test_strip_fillers_edge_cases():
    assert _strip_fillers("") == ""
    assert _strip_fillers("   ") == ""
    assert _strip_fillers(None) == ""


def test_compress_text_preserves_captions_and_handles_sentinel():
    # Normal caption with casing and 'like'
    assert _compress_text("A cloud that looks like a dragon in the sky") == "A cloud that looks like a dragon in the sky"

    # Multiple whitespace normalized
    assert _compress_text("  A   bright   red   apple  ") == "A bright red apple"

    # Empty and None return None (FIX-11 eliminated dummy failure sentinels)
    assert _compress_text("") is None
    assert _compress_text(None) is None
    assert _compress_text("   ") is None


def test_clean_audio_text():
    # Returns empty string for empty speech
    assert _clean_audio_text("") == ""
    assert _clean_audio_text(None) == ""
    assert _clean_audio_text("uh um") == ""

    # Normal spoken sentence
    assert _clean_audio_text("Um, we are presenting Maurya demo.") == "we are presenting Maurya demo."
    assert clean_audio_text.py_fn("Um, we are presenting Maurya demo.") == "we are presenting Maurya demo."


def test_compute_combined_caption_text():
    visual = "  A rabbit standing in a grassy meadow.  "
    transcript = {
        "segments": [
            {"start": 4.0, "end": 6.0, "text": "Um, look at that rabbit."},
            {"start": 10.0, "end": 12.0, "text": "Something else later."},
        ]
    }
    # Timestamp at 5.0 matches segment 1 (window 0.5s around 5.0)
    combined = _compute_combined_caption_text(visual, transcript, timestamp=5.0)
    assert combined == "A rabbit standing in a grassy meadow. Spoken audio: look at that rabbit."

    # Timestamp at 1.0 has no audio
    combined_silent = _compute_combined_caption_text(visual, transcript, timestamp=1.0)
    assert combined_silent == "A rabbit standing in a grassy meadow."
