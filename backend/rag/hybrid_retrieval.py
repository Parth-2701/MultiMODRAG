# backend/rag/hybrid_retrieval.py
import re
import os
from typing import Optional, List, Dict, Any
from rank_bm25 import BM25Okapi            # pip install rank-bm25
from snowballstemmer import stemmer        # pip install snowballstemmer

K_DENSE = int(os.environ.get("K_DENSE", 40))
K_SPARSE = int(os.environ.get("K_SPARSE", 40))
K_OUT = int(os.environ.get("K_OUT", 30))
RRF_K = 60
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "but", "what", "which", "who", "whom", "this", "that", "these",
    "those", "it", "its", "as", "by", "from", "about", "into", "than", "then", "there", "their", "how",
    "when", "where", "why", "can", "could", "would", "should", "show", "me", "tell", "please"
}
# NOTE: never filter by token length. Keep "cat", "car", "gpu", "rag", "ai".
_stem = stemmer("english").stemWord

def tokenize(text: str) -> list[str]:
    toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]
    return [_stem(t) for t in toks]

_cache: dict[str, tuple] = {}   # title -> (signature, bm25, rows)

def _lexical_index(frames, title: Optional[str], flt):
    select_cols = [frames.pos, frames.combined_caption, frames.video, frames.title]
    if hasattr(frames, "timestamp_sec"):
        select_cols.append(frames.timestamp_sec)
    rows = frames.where(flt).select(*select_cols).collect()
    sig = (len(rows), max((r["pos"] for r in rows), default=-1))
    cache_key = title if title is not None else "__all__"
    hit = _cache.get(cache_key)
    if hit and hit[0] == sig:
        return hit
    corpus = [tokenize(r["combined_caption"]) for r in rows]
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75) if corpus else None
    _cache[cache_key] = (sig, bm25, rows)
    return _cache[cache_key]

def rrf(ranked_id_lists: list[list[Any]], k: int = RRF_K) -> list[tuple[Any, float]]:
    score: dict[Any, float] = {}
    for lst in ranked_id_lists:                       # each list: best -> worst
        for rank, doc_id in enumerate(lst, start=1):  # rank starts at 1
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])

def hybrid_search(frames, title: Optional[str], query: str, fps: float, query_image: Optional[Any] = None) -> List[Dict[str, Any]]:
    flt = frames.combined_caption != None
    if title is not None:
        flt = flt & (frames.title == title)

    # 1) dense: bounded top-K, DESCENDING similarity
    # Use task-typed v2 index if enabled (FIX-13) and available on table, fallback to legacy
    use_v2 = os.environ.get("USE_TASK_EMBEDDINGS", "1") == "1"
    sim = None
    if use_v2 and hasattr(frames, "combined_emb"):
        try:
            sim = frames.combined_emb.similarity(string=query, idx="combined_caption_emb_v2")
        except Exception:
            sim = None

    if sim is None:
        try:
            sim = frames.combined_caption.similarity(string=query)
        except Exception:
            sim = frames.combined_caption.similarity(query)
    select_cols = [frames.pos, frames.combined_caption, frames.video, frames.title]
    if hasattr(frames, "timestamp_sec"):
        select_cols.append(frames.timestamp_sec)
    dense = (
        frames.where(flt)
        .order_by(sim, asc=False)
        .limit(K_DENSE)
        .select(*select_cols)
        .collect()
    )
    dense_ids = [r["pos"] for r in dense]

    # 2) sparse: true BM25 over ALL frames of this video (independent retriever)
    _, bm25, rows = _lexical_index(frames, title, flt)
    if not rows:
        return []

    q = tokenize(query)
    scores = bm25.get_scores(q) if (bm25 is not None and q) else []
    order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])[:K_SPARSE]
    sparse_ids = [rows[i]["pos"] for i in order]

    # 3) visual: direct CLIP visual ranking (FIX-18)
    visual_ids = []
    try:
        from backend.rag.visual_index import visual_rank
        visual_ids = visual_rank(frames, title, query_text=query, query_image=query_image, k=K_DENSE)
    except Exception:
        visual_ids = []

    # 4) fuse tri-modal ranked lists (dense + sparse + visual)
    ranked_lists = [l for l in [dense_ids, sparse_ids, visual_ids] if l]
    fused = rrf(ranked_lists, k=RRF_K)[:K_OUT] if ranked_lists else []
    by_pos = {r["pos"]: r for r in list(rows) + list(dense)}
    return [{
        "pos": pos,
        "timestamp_seconds": float(by_pos[pos]["timestamp_sec"]) if by_pos[pos].get("timestamp_sec") is not None else float(pos / fps),
        "caption": by_pos[pos].get("combined_caption", ""),
        "video_path": by_pos[pos].get("video", ""),
        "source_filename": by_pos[pos].get("title", title or "unknown.mp4"),
        "score": s,
    } for pos, s in fused if pos in by_pos]
