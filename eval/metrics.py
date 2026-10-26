"""
Evaluation metrics for Video RAG retrieval.
"""
from typing import List, Tuple, Optional, Dict, Any

def overlaps(a: Tuple[float, float], b: Tuple[float, float], tol: float = 3.0) -> bool:
    """
    Returns True if time interval a [start, end] overlaps time interval b with tolerance.
    """
    return a[0] - tol <= b[1] and b[0] - tol <= a[1]

def hit_rank(
    retrieved_spans: List[Dict[str, Any]],
    gold_spans: List[List[float]],
    tol: float = 3.0
) -> Optional[int]:
    """
    Finds the 1-based rank of the first retrieved span that overlaps any gold span.
    Returns None if no hit overlaps within tolerance.
    """
    if not gold_spans:
        # If no specific gold span (e.g. video-level summary), any valid retrieved evidence counts as rank 1
        return 1 if retrieved_spans else None

    for i, s in enumerate(retrieved_spans, start=1):
        s_start = float(s.get("start", s.get("timestamp_seconds", 0.0)))
        s_end = float(s.get("end", s_start + 1.0))
        for g in gold_spans:
            if overlaps((s_start, s_end), (float(g[0]), float(g[1])), tol=tol):
                return i
    return None

def recall_at_k(rank: Optional[int], k: int) -> float:
    """Returns 1.0 if rank is within top-k, else 0.0."""
    return 1.0 if rank is not None and rank <= k else 0.0

def mrr(rank: Optional[int]) -> float:
    """Returns reciprocal rank: 1.0 / rank, or 0.0 if not found."""
    return 0.0 if rank is None else 1.0 / rank
