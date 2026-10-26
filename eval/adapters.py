"""
Retrieval adapters for comparing baseline and enhanced RAG pipelines.
Signature: (title: str, query: str, k: int) -> list[dict]
Returned dict shape: {"start": float, "end": float, "score": float, "text": str}
"""
import re
from typing import List, Dict, Any, Optional
import pixeltable as pxt
from rank_bm25 import BM25Okapi

from backend.rag.hybrid_retrieval import tokenize, _lexical_index, hybrid_search
from backend.rag.spans import to_spans

FPS = 1.0

def dense_plus_proxy(title: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Original repo baseline logic:
    - Dense query without asc=False (defaults to asc=True in Pixeltable)
    - Naive substring keyword count with len(term) > 3
    - Magic constant rank: 15 - keyword_hits
    """
    frames = pxt.get_table("video_rag.frames")
    flt = (frames.combined_caption != None) & (frames.title == title)
    sim = frames.combined_caption.similarity(query)
    dense_results = (
        frames.where(flt)
        .order_by(sim)
        .select(frames.pos, frames.combined_caption)
        .limit(max(k * 2, 20))
        .collect()
    )
    query_terms = set(query.lower().replace("?", "").replace(".", "").split())
    ranked = []
    for i, row in enumerate(dense_results):
        caption_text = (row["combined_caption"] or "").lower()
        dense_rank = i + 1
        keyword_hits = sum(1 for term in query_terms if term in caption_text and len(term) > 3)
        sparse_rank = 15 - keyword_hits
        rrf_score = (1 / (60 + dense_rank)) + (1 / (60 + sparse_rank))
        pos = row["pos"]
        ranked.append({
            "start": round(pos / FPS, 2),
            "end": round((pos + 1) / FPS, 2),
            "score": rrf_score,
            "text": row["combined_caption"] or "",
        })
    ranked.sort(key=lambda x: -x["score"])
    return ranked[:k]

baseline_dense_proxy = dense_plus_proxy

def dense_only(title: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Dense retrieval alone with verified descending similarity.
    """
    frames = pxt.get_table("video_rag.frames")
    flt = (frames.combined_caption != None) & (frames.title == title)
    sim = frames.combined_caption.similarity(query)
    dense = (
        frames.where(flt)
        .order_by(sim, asc=False)
        .limit(k)
        .select(frames.pos, frames.combined_caption)
        .collect()
    )
    return [
        {
            "start": round(r["pos"] / FPS, 2),
            "end": round((r["pos"] + 1) / FPS, 2),
            "score": 1.0 / (i + 1),
            "text": r["combined_caption"] or "",
        }
        for i, r in enumerate(dense)
    ]

def bm25_only(title: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Independent sparse BM25 retrieval over frame captions.
    """
    frames = pxt.get_table("video_rag.frames")
    flt = (frames.combined_caption != None) & (frames.title == title)
    _, bm25, rows = _lexical_index(frames, title, flt)
    if not rows or bm25 is None:
        return []
    q_toks = tokenize(query)
    scores = bm25.get_scores(q_toks) if q_toks else []
    order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:k]
    return [
        {
            "start": round(rows[i]["pos"] / FPS, 2),
            "end": round((rows[i]["pos"] + 1) / FPS, 2),
            "score": float(scores[i]),
            "text": rows[i]["combined_caption"] or "",
        }
        for i in order
    ]

def hybrid_rrf_spans(title: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Our upgraded pipeline: Bounded Dense (asc=False) + True BM25 fused with RRF,
    followed by Temporal Span Merging and Deduplication (FIX-01 + FIX-04).
    """
    frames = pxt.get_table("video_rag.frames")
    ranked_hits = hybrid_search(frames, title, query, FPS)
    spans = to_spans(ranked_hits[:30], FPS)
    return [
        {
            "start": s["start"],
            "end": s["end"],
            "score": s["score"],
            "text": s["text"],
        }
        for s in spans[:k]
    ]

ADAPTERS = {
    "baseline_dense_proxy": dense_plus_proxy,
    "dense_only": dense_only,
    "bm25_only": bm25_only,
    "hybrid_rrf_spans": hybrid_rrf_spans,
}
