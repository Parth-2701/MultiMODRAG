"""
Unit tests for FIX-09: Reranker over fused candidates.
Verifies LLM reranking, stable tie-breaking, fail-open behavior, and bypass modes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.rag.rerank import rerank, _parse_scores, _get_caption, _get_timestamp


def test_cand_helpers():
    frame_dict = {"timestamp_seconds": 14.5, "caption": "Rabbit jumping in meadow"}
    assert _get_timestamp(frame_dict) == 14.5
    assert _get_caption(frame_dict) == "Rabbit jumping in meadow"

    span_dict = {"start": 8.0, "end": 12.0, "text": "Apple resting on the ground"}
    assert _get_timestamp(span_dict) == 8.0
    assert _get_caption(span_dict) == "Apple resting on the ground"

    empty_dict = {}
    assert _get_timestamp(empty_dict) == 0.0
    assert _get_caption(empty_dict) == ""


def test_parse_scores():
    raw_json = '[{"id": 0, "score": 1}, {"id": 1, "score": 3}, {"id": 2, "score": 0}]'
    scores = _parse_scores(raw_json, 3)
    assert scores == {0: 1.0, 1: 3.0, 2: 0.0}

    # Markdown fenced
    fenced = '```json\n[{"id": 0, "score": 2}]\n```'
    assert _parse_scores(fenced, 1) == {0: 2.0}

    # Malformed
    assert _parse_scores("not json", 2) == {}


@pytest.mark.anyio
async def test_rerank_mode_off(monkeypatch):
    monkeypatch.setenv("RERANK_MODE", "off")
    cands = [
        {"timestamp_seconds": 1.0, "caption": "Scene 1", "score": 0.9},
        {"timestamp_seconds": 2.0, "caption": "Scene 2", "score": 0.8},
        {"timestamp_seconds": 3.0, "caption": "Scene 3", "score": 0.7},
    ]
    res = await rerank(None, "test query", cands, n_in=10, n_out=2)
    # Mode off preserves exact first-stage order truncated to n_out
    assert len(res) == 2
    assert res[0]["caption"] == "Scene 1"
    assert res[1]["caption"] == "Scene 2"


@pytest.mark.anyio
async def test_rerank_small_candidate_list():
    cands = [{"timestamp_seconds": 1.0, "caption": "Only one"}]
    res = await rerank(None, "query", cands, n_out=5)
    assert res == cands


@pytest.mark.anyio
async def test_rerank_llm_reordering_and_stable_ties(monkeypatch):
    monkeypatch.setenv("RERANK_MODE", "llm")
    cands = [
        {"timestamp_seconds": 0.0, "caption": "Sun in the sky", "score": 0.95},       # id 0 -> score 0
        {"timestamp_seconds": 5.0, "caption": "Rabbit looking at butterfly", "score": 0.85}, # id 1 -> score 1
        {"timestamp_seconds": 7.0, "caption": "Apple falls and hits butterfly", "score": 0.75}, # id 2 -> score 3 (best)
        {"timestamp_seconds": 9.0, "caption": "Grass meadow background", "score": 0.65},  # id 3 -> score 0
    ]

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()

    mock_resp = MagicMock()
    mock_resp.text = '[{"id": 0, "score": 0}, {"id": 1, "score": 1}, {"id": 2, "score": 3}, {"id": 3, "score": 0}]'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    res = await rerank(mock_client, "When does the apple hit the butterfly?", cands, n_in=10, n_out=3)

    assert len(res) == 3
    # Candidate 2 had highest score (3), so it must be ranked first
    assert res[0]["timestamp_seconds"] == 7.0
    assert "Apple falls" in res[0]["caption"]

    # Candidate 1 had score 1, so it is second
    assert res[1]["timestamp_seconds"] == 5.0

    # Candidate 0 and Candidate 3 both had score 0. Stable tie-break must retain Candidate 0 (earlier first-stage index)
    assert res[2]["timestamp_seconds"] == 0.0


@pytest.mark.anyio
async def test_rerank_fail_open_on_error(monkeypatch):
    monkeypatch.setenv("RERANK_MODE", "llm")
    cands = [
        {"timestamp_seconds": 1.0, "caption": "Cand 1"},
        {"timestamp_seconds": 2.0, "caption": "Cand 2"},
    ]

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API timeout"))

    # Must fail open gracefully without raising exception
    res = await rerank(mock_client, "query", cands, n_in=10, n_out=2)
    assert len(res) == 2
    assert res[0]["caption"] == "Cand 1"
    assert res[1]["caption"] == "Cand 2"
