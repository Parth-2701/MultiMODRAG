import pytest
from eval.metrics import overlaps, hit_rank, recall_at_k, mrr
from eval.run_eval import percentile

def test_overlaps():
    # Exactly overlapping
    assert overlaps((10.0, 20.0), (15.0, 25.0), tol=0.0) is True
    # Non-overlapping without tolerance
    assert overlaps((10.0, 15.0), (20.0, 25.0), tol=0.0) is False
    # Overlapping within 5.0 tolerance (gap is 5.0)
    assert overlaps((10.0, 15.0), (20.0, 25.0), tol=5.0) is True
    # Touching boundary
    assert overlaps((10.0, 15.0), (15.0, 20.0), tol=0.0) is True

def test_hit_rank_handmade_lists():
    gold = [[60.0, 75.0]]

    # Case 1: First item hits
    hits1 = [
        {"start": 62.0, "end": 68.0, "score": 0.9},
        {"start": 10.0, "end": 15.0, "score": 0.5},
    ]
    assert hit_rank(hits1, gold, tol=1.0) == 1

    # Case 2: Third item hits
    hits2 = [
        {"start": 10.0, "end": 15.0, "score": 0.9},
        {"start": 20.0, "end": 25.0, "score": 0.8},
        {"start": 70.0, "end": 74.0, "score": 0.7},
    ]
    assert hit_rank(hits2, gold, tol=1.0) == 3

    # Case 3: No item hits
    hits3 = [
        {"start": 10.0, "end": 15.0, "score": 0.9},
        {"start": 20.0, "end": 25.0, "score": 0.8},
    ]
    assert hit_rank(hits3, gold, tol=1.0) is None

    # Case 4: Multiple gold spans
    multi_gold = [[10.0, 15.0], [80.0, 90.0]]
    hits4 = [{"start": 82.0, "end": 85.0}]
    assert hit_rank(hits4, multi_gold) == 1

def test_recall_at_k():
    assert recall_at_k(1, 1) == 1.0
    assert recall_at_k(1, 5) == 1.0
    assert recall_at_k(3, 1) == 0.0
    assert recall_at_k(3, 3) == 1.0
    assert recall_at_k(5, 3) == 0.0
    assert recall_at_k(None, 10) == 0.0

def test_mrr():
    assert mrr(1) == 1.0
    assert mrr(2) == 0.5
    assert mrr(4) == 0.25
    assert mrr(10) == 0.1
    assert mrr(None) == 0.0

def test_percentile():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 0) == 10.0
    assert percentile(vals, 100) == 50.0
