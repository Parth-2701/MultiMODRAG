# backend/rag/rerank.py
import os
import json
import re
from typing import List, Dict, Any, Optional

MODE = os.environ.get("RERANK_MODE", "llm")  # "off" | "llm" | "cross_encoder"
N_IN = int(os.environ.get("RERANK_IN", 30))
N_OUT = int(os.environ.get("RERANK_OUT", 15))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "gemini-2.5-flash")
_ce = None


def _get_caption(cand: Dict[str, Any]) -> str:
    """Helper to extract text or caption field from candidate dict."""
    return str(cand.get("caption") or cand.get("text") or "").strip()


def _get_timestamp(cand: Dict[str, Any]) -> float:
    """Helper to extract timestamp from candidate dict."""
    ts = cand.get("timestamp_seconds")
    if ts is None:
        ts = cand.get("start", 0.0)
    try:
        return float(ts)
    except (ValueError, TypeError):
        return 0.0


def _parse_scores(raw_text: str, n: int) -> Dict[int, float]:
    """Parses JSON score mapping from LLM text output."""
    if not raw_text:
        return {}
    cleaned = re.sub(r"^```(?:json)?", "", raw_text.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"(\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                pass

    by_id = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item and "score" in item:
                try:
                    by_id[int(item["id"])] = float(item["score"])
                except (ValueError, TypeError):
                    pass
    return by_id


async def rerank(
    client: Any,
    query: str,
    cands: List[Dict[str, Any]],
    n_in: Optional[int] = None,
    n_out: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Second-stage cross-attention / LLM reranking over first-stage candidates.
    Evaluates passages directly against query and sorts with stable tie-breaking.
    Fails open (returns first-stage order) if disabled, client missing, or on error.
    """
    if not cands or not query:
        return cands or []

    limit_in = n_in if n_in is not None else N_IN
    limit_out = n_out if n_out is not None else N_OUT

    candidates = list(cands[:limit_in])
    mode = os.environ.get("RERANK_MODE", MODE).lower()

    if mode == "off" or len(candidates) <= limit_out:
        return candidates[:limit_out]

    try:
        if mode == "cross_encoder":
            global _ce
            if _ce is None:
                from sentence_transformers import CrossEncoder
                _ce = CrossEncoder("BAAI/bge-reranker-base")
            pairs = [(query, _get_caption(c)[:1000]) for c in candidates]
            scores = list(_ce.predict(pairs))
        else:
            # LLM Reranking mode
            if client is None:
                return candidates[:limit_out]

            listing = "\n".join(
                f"[{i}] ({_get_timestamp(c):.0f}s) {_get_caption(c)[:350]}"
                for i, c in enumerate(candidates)
            )
            prompt = f"""Query: {query}
Rate how directly each video passage helps answer the query:
0: irrelevant or off-topic
1: related general context
2: relevant information
3: directly and precisely answers the query

Return strictly a JSON array of objects for every id:
[{{"id": int, "score": 0-3}}, ...]

Passages:
{listing}"""

            raw_out = ""
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                resp = await client.aio.models.generate_content(
                    model=RERANK_MODEL,
                    contents=prompt,
                    config={"temperature": 0.0, "response_mime_type": "application/json"},
                )
                raw_out = (resp.text or "").strip()
            elif hasattr(client, "models"):
                resp = client.models.generate_content(
                    model=RERANK_MODEL,
                    contents=prompt,
                    config={"temperature": 0.0, "response_mime_type": "application/json"},
                )
                raw_out = (resp.text or "").strip()
            else:
                return candidates[:limit_out]

            by_id = _parse_scores(raw_out, len(candidates))
            # Default to original position rank if unrated: (len - i) / len to break ties
            scores = [by_id.get(i, 0.0) for i in range(len(candidates))]

        # Stable sort: high score first; tie breaks with earlier first-stage index (i)
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order][:limit_out]

    except Exception as e:
        print(f"[rerank] Reranking exception, failing open to first stage: {e}")
        return candidates[:limit_out]
