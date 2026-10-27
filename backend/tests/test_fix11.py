"""Unit tests for FIX-11: Caption cascade hygiene, circuit breaker, no failure sentinels.

Verifies:
1. Circuit breaker tracking per model (404 permanent drop, 429 cooldown, consecutive failure backoff).
2. Skipping exhausted/dead models without wasting network round trips.
3. Preservation of all 12+ cascade models in GEMINI_CAPTION_MODELS.
4. Clean elimination of 'unlabeled visual content' sentinels across Florence-2,
   gemini_caption, compress_text, and compute_combined_caption_text.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

import backend.schema as schema
from backend.schema import (
    GEMINI_CAPTION_MODELS,
    _model_state,
    _is_model_usable,
    _record_success,
    _record_failure,
    _run_florence,
    _compress_text,
    _compute_combined_caption_text,
    gemini_caption,
)


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker state before and after each test."""
    for m in GEMINI_CAPTION_MODELS:
        _model_state[m] = {"until": 0.0, "consecutive_fails": 0}
    yield
    for m in GEMINI_CAPTION_MODELS:
        _model_state[m] = {"until": 0.0, "consecutive_fails": 0}


def test_gemini_models_intact():
    """Resume claim verification: ensure all 12+ Gemini models remain configured in cascade."""
    assert len(GEMINI_CAPTION_MODELS) >= 12, (
        f"Expected at least 12 cascade models for 12x free-tier throughput claim, "
        f"found {len(GEMINI_CAPTION_MODELS)}: {GEMINI_CAPTION_MODELS}"
    )


def test_circuit_breaker_404_permanent_drop():
    """404 or NOT_FOUND models must be dropped permanently (until = inf)."""
    model = GEMINI_CAPTION_MODELS[0]
    assert _is_model_usable(model) is True

    _record_failure(model, RuntimeError("404 models/gemini-old is not found for API version v1beta"))
    assert _model_state[model]["until"] == float("inf")
    assert _is_model_usable(model) is False


def test_circuit_breaker_429_cooldown():
    """429 or RESOURCE_EXHAUSTED models must be cooled down for ~60s."""
    model = GEMINI_CAPTION_MODELS[0]
    start_time = time.time()
    _record_failure(model, RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
    
    assert _model_state[model]["until"] >= start_time + 59.0
    assert _is_model_usable(model) is False

    # Simulate fast-forwarding time past cooldown
    _model_state[model]["until"] = time.time() - 1.0
    assert _is_model_usable(model) is True


def test_circuit_breaker_consecutive_failures():
    """3 consecutive non-rate-limit failures must trigger a 5-minute cooldown."""
    model = GEMINI_CAPTION_MODELS[1]
    
    _record_failure(model, ValueError("Bad connection"))
    assert _model_state[model]["consecutive_fails"] == 1
    assert _is_model_usable(model) is True

    _record_failure(model, ValueError("Bad connection 2"))
    assert _model_state[model]["consecutive_fails"] == 2
    assert _is_model_usable(model) is True

    start_time = time.time()
    _record_failure(model, ValueError("Bad connection 3"))
    assert _model_state[model]["consecutive_fails"] == 0
    assert _model_state[model]["until"] >= start_time + 290.0
    assert _is_model_usable(model) is False


def test_circuit_breaker_success_resets_failures():
    """Successful caption generation resets consecutive failure count."""
    model = GEMINI_CAPTION_MODELS[0]
    _record_failure(model, ValueError("Transient glitch"))
    assert _model_state[model]["consecutive_fails"] == 1

    _record_success(model)
    assert _model_state[model]["consecutive_fails"] == 0


def test_cascade_skips_cooling_models():
    """gemini_caption must skip models in cooldown and try the next usable model."""
    img = Image.new("RGB", (64, 64), color="blue")
    
    # Mark first model as cooling down
    first_model = GEMINI_CAPTION_MODELS[0]
    second_model = GEMINI_CAPTION_MODELS[1]
    _model_state[first_model]["until"] = time.time() + 60.0

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"summary": "A blue square frame", "actions": [], "scene_type": "graphic", "detected_text": []}'
    
    # Mock client generate_content
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(schema, "_genai_client", mock_client):
        # Call underlying py_fn or udf
        fn = getattr(gemini_caption, "py_fn", gemini_caption)
        result = fn(img)

        # Ensure first_model was skipped and second_model was invoked
        called_models = [call.kwargs.get("model") for call in mock_client.models.generate_content.call_args_list]
        assert first_model not in called_models
        assert second_model in called_models
        assert "A blue square frame" in result


def test_no_failure_sentinels_florence():
    """Florence-2 must return None (not 'unlabeled visual content') on failure or None frame."""
    assert _run_florence(None) is None

    with patch.object(schema, "_get_florence", side_effect=RuntimeError("CUDA OOM")):
        img = Image.new("RGB", (10, 10))
        assert _run_florence(img) is None


def test_no_failure_sentinels_compress_text():
    """_compress_text must return None on empty or whitespace strings."""
    assert _compress_text(None) is None
    assert _compress_text("") is None
    assert _compress_text("   \n\t  ") is None
    assert _compress_text("Valid caption text") == "Valid caption text"


def test_no_failure_sentinels_gemini_caption():
    """gemini_caption must return None when frame is None."""
    fn = getattr(gemini_caption, "py_fn", gemini_caption)
    assert fn(None) is None


def test_compute_combined_caption_text_modalities():
    """_compute_combined_caption_text must handle all 4 modality combinations without sentinels."""
    transcript = {
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "Hello world from audio"}
        ]
    }

    # 1. Both visual and speech
    both = _compute_combined_caption_text("A man on stage.", transcript, 2.0)
    assert both == "A man on stage. Spoken audio: Hello world from audio"

    # 2. Visual only (silent video/frame)
    visual_only = _compute_combined_caption_text("A silent landscape.", transcript, 10.0)
    assert visual_only == "A silent landscape."

    # 3. Audio only (visual caption failed or None)
    audio_only = _compute_combined_caption_text(None, transcript, 2.0)
    assert audio_only == "Spoken audio: Hello world from audio"

    # 4. Neither visual nor speech
    neither = _compute_combined_caption_text(None, transcript, 10.0)
    assert neither is None

    neither_empty_str = _compute_combined_caption_text("", transcript, 10.0)
    assert neither_empty_str is None
