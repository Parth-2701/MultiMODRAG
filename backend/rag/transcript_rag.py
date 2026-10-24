# backend/rag/transcript_rag.py
import os
import re
from typing import Optional, List, Dict, Any
import pixeltable as pxt
from pixeltable.functions.gemini import embed_content
from rank_bm25 import BM25Okapi

TARGET_SEC = float(os.environ.get("TX_CHUNK_SEC", 45))
MAX_WORDS = int(os.environ.get("TX_CHUNK_WORDS", 140))
OVERLAP = int(os.environ.get("TX_OVERLAP_SEGS", 1))
TOP_K = int(os.environ.get("TX_TOP_K", 4))
FULL_MAX_TOKENS = int(os.environ.get("TX_FULL_MAX_TOKENS", 3000))   # short videos: send all
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did", "of", "in", "on", "at",
    "to", "for", "with", "and", "or", "but", "what", "which", "who", "whom", "this", "that", "these",
    "those", "it", "its", "as", "by", "from", "about", "into", "than", "then", "there", "their", "how",
    "when", "where", "why", "can", "could", "would", "should", "show", "me", "tell", "please"
}

try:
    from snowballstemmer import stemmer
    _stem = stemmer("english").stemWord
except Exception:
    _stem = lambda x: x

def tokenize(text: str) -> list[str]:
    toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]
    return [_stem(t) for t in toks]

def rrf(ranked_id_lists: list[list[Any]], k: int = RRF_K) -> list[tuple[Any, float]]:
    score: dict[Any, float] = {}
    for lst in ranked_id_lists:
        for rank, doc_id in enumerate(lst, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])

def mmss(sec: float) -> str:
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"

def _make(segs: list[dict]) -> dict:
    return {
        "start": segs[0]["start"],
        "end": segs[-1]["end"],
        "text": " ".join(s["text"].strip() for s in segs).strip()
    }

def chunk_segments(segments: list[dict]) -> list[dict]:
    chunks, cur, last_emitted_tail = [], [], 0
    for seg in segments:
        cur.append(seg)
        dur = cur[-1]["end"] - cur[0]["start"]
        words = sum(len(s["text"].split()) for s in cur)
        if dur >= TARGET_SEC or words >= MAX_WORDS:
            chunks.append(_make(cur))
            cur = cur[-OVERLAP:] if OVERLAP else []
            last_emitted_tail = len(cur)
    if len(cur) > last_emitted_tail:                 # trailing material not yet emitted
        chunks.append(_make(cur))
    return chunks

def ensure_chunks(title: str):
    try:
        tbl = pxt.create_table(
            "video_rag.transcript_chunks",
            {"title": pxt.String, "start": pxt.Float, "end": pxt.Float, "text": pxt.String},
            if_exists="ignore"
        )
        tbl.add_embedding_index(
            "text",
            idx_name="tx_idx",
            embedding=embed_content.using(model="gemini-embedding-001"),
            if_exists="ignore"
        )
    except Exception:
        try:
            tbl = pxt.get_table("video_rag.transcript_chunks")
        except Exception:
            return None

    if tbl.where(tbl.title == title).count() > 0:
        return tbl

    try:
        videos = pxt.get_table("video_rag.videos")
        row = videos.where(videos.title == title).select(videos.transcript).collect()
    except Exception:
        return tbl

    segs = (
        (row[0]["transcript"] or {}).get("segments", [])
        if len(row) and isinstance(row[0]["transcript"], dict)
        else []
    )
    chunks = chunk_segments(segs)
    if chunks:
        tbl.insert([{"title": title, **c} for c in chunks])
    return tbl

def _fmt(c: dict) -> str:
    return f"[{mmss(c['start'])}–{mmss(c['end'])}] {c['text']}"

def transcript_context(title: Optional[str], query: str) -> str:
    if not title:
        return ""
    tbl = ensure_chunks(title)
    if tbl is None:
        return ""

    rows = (
        tbl.where(tbl.title == title)
        .select(tbl.start, tbl.end, tbl.text)
        .order_by(tbl.start)
        .collect()
    )
    if len(rows) == 0:
        return ""

    total_tokens = int(sum(len(r["text"].split()) for r in rows) * 1.3)
    if total_tokens <= FULL_MAX_TOKENS:               # short video: whole transcript, WITH timestamps
        return "\n".join(_fmt(r) for r in rows)

    # Long video: dense + BM25 hybrid search over chunks
    sim = tbl.text.similarity(query)
    dense = (
        tbl.where(tbl.title == title)
        .order_by(sim, asc=False)
        .limit(20)
        .select(tbl.start, tbl.end, tbl.text)
        .collect()
    )

    bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])
    q_toks = tokenize(query)
    sc = bm25.get_scores(q_toks) if q_toks else []
    sparse = [rows[i] for i in sorted(range(len(rows)), key=lambda i: -sc[i]) if sc[i] > 0][:20]

    fused = rrf([[r["start"] for r in dense], [r["start"] for r in sparse]], k=RRF_K)[:TOP_K]
    keep = {s for s, _ in fused}
    picked = sorted((r for r in rows if r["start"] in keep), key=lambda r: r["start"])   # chronological
    return "\n".join(_fmt(r) for r in picked)
