"""Unit tests for FIX-12: Whisper robustness and distinguishing silent from failed audio.

Verifies:
1. Multilingual default model and VAD/repetition parameter configuration.
2. Status taxonomy: 'no_audio', 'ok', 'silent', 'failed'.
3. Pruning of hallucinations based on no_speech_prob and avg_logprob.
4. Honest prompt grounding in transcript_context for silent vs failed vs no_audio.
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import backend.schema as schema
from backend.schema import (
    WHISPER_MODEL_SIZE,
    _run_transcribe_audio,
)
from backend.rag.transcript_rag import transcript_context


def test_whisper_config_defaults():
    """Verify Whisper model size is multilingual 'base' by default."""
    assert WHISPER_MODEL_SIZE in ("base", "tiny", "small")


def test_transcribe_audio_no_audio():
    """Missing or null audio input returns status 'no_audio'."""
    res = _run_transcribe_audio(None)
    assert res["status"] == "no_audio"
    assert res["text"] == ""
    assert res["segments"] == []


def test_transcribe_audio_ok_english():
    """Valid speech produces status 'ok', correct segments, and concatenated text."""
    mock_seg1 = SimpleNamespace(start=0.0, end=3.5, text="Welcome to the presentation.", no_speech_prob=0.02, avg_logprob=-0.2)
    mock_seg2 = SimpleNamespace(start=4.0, end=7.2, text="Today we discuss RAG systems.", no_speech_prob=0.05, avg_logprob=-0.15)
    mock_info = SimpleNamespace(language="en", language_probability=0.99)

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_seg1, mock_seg2], mock_info)

    with patch.object(schema, "_get_whisper", return_value=mock_whisper):
        res = _run_transcribe_audio("dummy_audio")

        # Verify transcription parameters include VAD and condition_on_previous_text=False
        args, kwargs = mock_whisper.transcribe.call_args
        assert kwargs.get("beam_size") == 5
        assert kwargs.get("vad_filter") is True
        assert kwargs.get("condition_on_previous_text") is False
        assert kwargs.get("vad_parameters", {}).get("min_silence_duration_ms") == 500

        assert res["status"] == "ok"
        assert res["language"] == "en"
        assert res["language_probability"] == 0.99
        assert res["text"] == "Welcome to the presentation. Today we discuss RAG systems."
        assert len(res["segments"]) == 2
        assert res["segments"][0] == {"start": 0.0, "end": 3.5, "text": "Welcome to the presentation."}


def test_transcribe_audio_ok_multilingual():
    """Multilingual speech is transcribed in its native language with proper language code."""
    mock_seg = SimpleNamespace(start=1.0, end=5.0, text="Hola a todos y bienvenidos.", no_speech_prob=0.01, avg_logprob=-0.1)
    mock_info = SimpleNamespace(language="es", language_probability=0.96)

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([mock_seg], mock_info)

    with patch.object(schema, "_get_whisper", return_value=mock_whisper):
        res = _run_transcribe_audio("dummy_spanish_audio")
        assert res["status"] == "ok"
        assert res["language"] == "es"
        assert res["text"] == "Hola a todos y bienvenidos."


def test_transcribe_audio_hallucination_filtering():
    """Segments with no_speech_prob > 0.6 or avg_logprob < -1.0 are dropped."""
    real_seg = SimpleNamespace(start=0.0, end=2.0, text="Legitimate sentence.", no_speech_prob=0.05, avg_logprob=-0.2)
    hallucination_silence = SimpleNamespace(start=2.5, end=6.0, text="Thank you for watching.", no_speech_prob=0.85, avg_logprob=-0.3)
    hallucination_low_prob = SimpleNamespace(start=6.5, end=9.0, text="Gibberish acoustic noise.", no_speech_prob=0.1, avg_logprob=-1.7)
    mock_info = SimpleNamespace(language="en", language_probability=0.95)

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([real_seg, hallucination_silence, hallucination_low_prob], mock_info)

    with patch.object(schema, "_get_whisper", return_value=mock_whisper):
        res = _run_transcribe_audio("dummy_audio")
        assert res["status"] == "ok"
        assert len(res["segments"]) == 1
        assert res["segments"][0]["text"] == "Legitimate sentence."
        assert "Thank you for watching" not in res["text"]
        assert "Gibberish" not in res["text"]


def test_transcribe_audio_silent_video():
    """Audio without speech yields status 'silent' and empty text."""
    mock_info = SimpleNamespace(language="en", language_probability=0.4)

    mock_whisper = MagicMock()
    # Entire segment iteration is empty or all pruned
    phantom_seg = SimpleNamespace(start=0.0, end=5.0, text="Subtitles by...", no_speech_prob=0.95, avg_logprob=-1.5)
    mock_whisper.transcribe.return_value = ([phantom_seg], mock_info)

    with patch.object(schema, "_get_whisper", return_value=mock_whisper):
        res = _run_transcribe_audio("dummy_silent_audio")
        assert res["status"] == "silent"
        assert res["text"] == ""
        assert res["segments"] == []


def test_transcribe_audio_failed_on_exception():
    """Exceptions return status 'failed' with error string rather than silent success."""
    mock_whisper = MagicMock()
    mock_whisper.transcribe.side_effect = RuntimeError("Corrupt audio stream / decoder error")

    with patch.object(schema, "_get_whisper", return_value=mock_whisper):
        res = _run_transcribe_audio("corrupted_audio")
        assert res["status"] == "failed"
        assert "Corrupt audio stream" in res["error"]
        assert res["text"] == ""
        assert res["segments"] == []


def test_transcript_context_grounded_wording():
    """transcript_context outputs clear honest notices distinguishing silent from failed from no_audio."""
    # 1. Silent video
    with patch("backend.rag.transcript_rag.get_transcript_metadata", return_value={"status": "silent"}):
        ctx = transcript_context("sample.mp4", "What did he say?")
        assert ctx == "(this video has no detectable speech)"

    # 2. Failed transcription
    with patch("backend.rag.transcript_rag.get_transcript_metadata", return_value={"status": "failed", "error": "Decoder error"}):
        ctx = transcript_context("sample.mp4", "What did he say?")
        assert ctx == "(speech transcription failed; answer from visuals only and say so)"

    # 3. No audio track
    with patch("backend.rag.transcript_rag.get_transcript_metadata", return_value={"status": "no_audio"}):
        ctx = transcript_context("sample.mp4", "What did he say?")
        assert ctx == "(no audio track present in this video)"
