# backend/rag/openrouter.py
"""
OpenRouter Free Vision Model Fallback for Video RAG Frame Captioning.

Tier 2 Multimodal Captioning Engine:
- Automatically discovers and ranks currently active free vision models from OpenRouter.
- Uses OpenRouter's native multi-model fallback array (`models: [...]`) for automatic server-side failover.
- Built-in circuit breaker to prevent latency degradation when rate limits (429) or provider outages occur.
- Compresses and encodes frames as JPEG data URIs to minimize network payload.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", 15.0))
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://github.com/Parth-2701/MultiMODRAG")
OPENROUTER_APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "MultiMODRAG Video RAG")

# Curated priority ranking for free vision models (best visual OCR and reasoning first)
PREFERRED_FREE_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",  # Built-in meta-router: auto-routes to any free model with vision support
]

# Cache discovered models for 1 hour to avoid network overhead during video ingestion
_cached_free_models: list[str] = []
_cache_timestamp: float = 0.0
CACHE_TTL_SEC: float = 3600.0

# Circuit breaker state for OpenRouter endpoint
_circuit_state: dict[str, Any] = {
    "until": 0.0,
    "consecutive_fails": 0,
    "last_error": None,
}


def is_openrouter_configured() -> bool:
    """Returns True if an OpenRouter API key is set in environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    return bool(key and key.strip() and key != "your_openrouter_api_key_here")


def is_circuit_available() -> bool:
    """Checks whether the OpenRouter circuit-breaker cooldown has expired."""
    return time.time() >= _circuit_state["until"]


def record_success() -> None:
    """Resets consecutive failure counter on successful caption generation."""
    _circuit_state["consecutive_fails"] = 0
    _circuit_state["last_error"] = None


def record_failure(error_msg: str, status_code: Optional[int] = None) -> None:
    """Applies progressive circuit-breaker cooldowns on failure."""
    _circuit_state["last_error"] = error_msg
    if status_code == 401 or status_code == 403:
        # Invalid / missing key: disable for this server session
        _circuit_state["until"] = float("inf")
        print(f"[openrouter] Authentication failure ({status_code}). OpenRouter disabled for this session.")
    elif status_code == 429:
        # Rate-limited: 60s cooldown
        _circuit_state["until"] = time.time() + 60.0
        print("[openrouter] Rate-limited (429). Cooled down for 60s.")
    else:
        _circuit_state["consecutive_fails"] += 1
        if _circuit_state["consecutive_fails"] >= 3:
            _circuit_state["until"] = time.time() + 180.0
            _circuit_state["consecutive_fails"] = 0
            print("[openrouter] 3 consecutive failures. Cooled down for 3 minutes.")


def get_free_vision_models() -> list[str]:
    """
    Returns prioritized list of free vision model slugs.
    Queries OpenRouter's live /models catalog when cache expires.
    """
    global _cached_free_models, _cache_timestamp
    now = time.time()

    if _cached_free_models and (now - _cache_timestamp < CACHE_TTL_SEC):
        return list(_cached_free_models)

    discovered: list[str] = []
    try:
        url = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"
        resp = requests.get(
            url,
            headers={"User-Agent": "MultiMODRAG/1.0"},
            timeout=4.0,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for m in data:
                m_id = m.get("id", "")
                pricing = m.get("pricing", {})
                arch = m.get("architecture", {})
                inputs = arch.get("input_modalities", [])

                is_free = (pricing.get("prompt") == "0" or ":free" in m_id)
                has_vision = "image" in inputs
                # Exclude safety/moderation-only models
                is_usable = not any(k in m_id.lower() for k in ["safety", "guard", "moderation"])

                if is_free and has_vision and is_usable:
                    discovered.append(m_id)
    except Exception as e:
        # Fail open to preferred static list if model catalog query fails
        print(f"[openrouter] Notice: Live model catalog query skipped ({e}), using curated priority list.")

    # Sort discovered models prioritizing proven high-capability models
    ordered: list[str] = []
    for pref in PREFERRED_FREE_VISION_MODELS:
        if pref in discovered and pref not in ordered:
            ordered.append(pref)
        elif pref == "openrouter/free" and "openrouter/free" not in ordered:
            # openrouter/free is always appended as the ultimate router fallback
            ordered.append(pref)

    # Append any other newly discovered free vision models
    for disc in discovered:
        if disc not in ordered:
            ordered.append(disc)

    if not ordered:
        ordered = list(PREFERRED_FREE_VISION_MODELS)

    _cached_free_models = ordered
    _cache_timestamp = now
    return list(_cached_free_models)


def encode_frame_to_data_uri(frame: Image.Image, max_dim: int = 1024, quality: int = 85) -> str:
    """
    Resizes and encodes a PIL Image to a compact base64 JPEG data URI.
    Keeps payload small (~80-150KB) for rapid network transfer to OpenRouter.
    """
    img = frame
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def caption_frame_openrouter(frame: Image.Image, prompt: str) -> Optional[str]:
    """
    Submits a frame to OpenRouter's free vision models with automatic fallback.
    Returns the generated caption string or None if unserviceable.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or not key.strip() or key == "your_openrouter_api_key_here":
        return None

    if not is_circuit_available():
        return None

    try:
        models_to_try = get_free_vision_models()
        image_uri = encode_frame_to_data_uri(frame)

        # OpenRouter supports a `models` array for automatic server-side failover
        payload = {
            "models": models_to_try,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }

        headers = {
            "Authorization": f"Bearer {key.strip()}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_TITLE,
            "Content-Type": "application/json",
        }

        url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    model_used = data.get("model", models_to_try[0])
                    print(f"✨ [openrouter] Frame captioned successfully using '{model_used}'")
                    record_success()
                    return content
            print("[openrouter] Empty choices received from OpenRouter.")
            record_failure("Empty choices returned", 200)
            return None

        err_text = resp.text[:300]
        print(f"[openrouter] HTTP {resp.status_code} error: {err_text}")
        record_failure(err_text, resp.status_code)
        return None

    except requests.exceptions.Timeout:
        print(f"[openrouter] Request timed out after {OPENROUTER_TIMEOUT}s.")
        record_failure("Request timeout", None)
        return None
    except Exception as e:
        print(f"[openrouter] Error during caption call: {e}")
        record_failure(str(e), None)
        return None
