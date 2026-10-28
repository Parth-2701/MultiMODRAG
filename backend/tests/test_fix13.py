"""Unit tests for FIX-13: Task-typed, normalized, smaller embeddings (batched).

Verifies:
1. L2 normalization produces true unit vectors (||v||_2 == 1.0).
2. embed_docs batches requests (e.g. 250 texts -> 3 API calls of size 100).
3. Correct asymmetric task types:
   - RETRIEVAL_DOCUMENT for ingested text
   - RETRIEVAL_QUERY for search queries
4. Truncated dimensionality (default 768).
5. Hybrid search selection of v2 index with fallback to legacy.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from backend.rag.embeddings import (
    DIM,
    _norm,
    embed_docs,
    _embed_query,
)
import backend.rag.hybrid_retrieval as hr


def test_norm_unit_vector():
    """_norm produces unit-length vectors with float32 precision."""
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    normed = _norm(v)
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-5)
    assert normed.dtype == np.float32

    # Zero vector does not divide by zero or raise
    zero = np.zeros(10, dtype=np.float32)
    normed_zero = _norm(zero)
    assert np.all(normed_zero == 0.0)


def test_embed_docs_batching_and_task_type():
    """embed_docs batches texts up to batch_size and passes RETRIEVAL_DOCUMENT."""
    mock_client = MagicMock()

    # Create dummy response generator for batches
    def fake_embed_content(*args, **kwargs):
        contents = kwargs.get("contents", [])
        config = kwargs.get("config")
        assert config.task_type == "RETRIEVAL_DOCUMENT"
        assert config.output_dimensionality == DIM
        embs = [
            SimpleNamespace(values=list(np.random.randn(DIM).astype(float)))
            for _ in contents
        ]
        return SimpleNamespace(embeddings=embs)

    mock_client.models.embed_content.side_effect = fake_embed_content

    # 250 texts with batch=100 -> exactly 3 calls (100, 100, 50)
    texts = [f"Frame description number {i}" for i in range(250)]
    vectors = embed_docs(mock_client, texts, batch=100, dim=DIM)

    assert len(vectors) == 250
    assert mock_client.models.embed_content.call_count == 3

    # All returned vectors must be normalized unit vectors of dimension DIM
    for vec in vectors:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (DIM,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_embed_query_task_type():
    """_embed_query uses RETRIEVAL_QUERY and outputs unit normalized vector."""
    mock_client = MagicMock()
    fake_vals = list(np.random.randn(DIM).astype(float))
    mock_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=fake_vals)]
    )

    q_vec = _embed_query("find the red sports car", client=mock_client, dim=DIM)

    assert mock_client.models.embed_content.call_count == 1
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs.get("contents") == "find the red sports car"
    assert call_kwargs.get("config").task_type == "RETRIEVAL_QUERY"
    assert call_kwargs.get("config").output_dimensionality == DIM

    assert isinstance(q_vec, np.ndarray)
    assert q_vec.shape == (DIM,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)


def test_embed_docs_empty_and_failures():
    """embed_docs handles empty inputs and failures gracefully without crashing."""
    assert embed_docs(None, []) == []

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("API Quota exceeded")

    # When API fails, returns None for each item (fail-open)
    res = embed_docs(mock_client, ["Caption A", "Caption B"])
    assert len(res) == 2
    assert res[0] is None
    assert res[1] is None


def test_hybrid_search_selects_v2_index():
    """hybrid_search uses combined_caption_emb_v2 when USE_TASK_EMBEDDINGS=1."""
    mock_frames = MagicMock()
    mock_frames.combined_caption = MagicMock()
    mock_frames.combined_emb = MagicMock()

    # Mock lexical index returning empty to isolate dense retrieval
    with patch("backend.rag.hybrid_retrieval._lexical_index", return_value=(None, None, [])):
        with patch.dict(os.environ, {"USE_TASK_EMBEDDINGS": "1"}):
            hr.hybrid_search(mock_frames, "test.mp4", "sample query", fps=1.0)

            # Check that similarity was called on combined_emb with string modality and idx
            mock_frames.combined_emb.similarity.assert_called_once_with(
                string="sample query", idx="combined_caption_emb_v2"
            )


def test_hybrid_search_falls_back_when_flag_disabled():
    """hybrid_search falls back to combined_caption when USE_TASK_EMBEDDINGS=0."""
    mock_frames = MagicMock()
    mock_frames.combined_caption = MagicMock()
    mock_frames.combined_emb = MagicMock()

    with patch("backend.rag.hybrid_retrieval._lexical_index", return_value=(None, None, [])):
        with patch.dict(os.environ, {"USE_TASK_EMBEDDINGS": "0"}):
            hr.hybrid_search(mock_frames, "test.mp4", "sample query", fps=1.0)

            # Check that similarity was called on combined_caption
            mock_frames.combined_emb.similarity.assert_not_called()
            mock_frames.combined_caption.similarity.assert_called_once()
