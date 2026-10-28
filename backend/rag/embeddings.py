"""backend/rag/embeddings.py: Task-typed, normalized, batched embeddings (FIX-13).

Provides:
- Task-typed embedding generation:
  - RETRIEVAL_DOCUMENT for stored video frame captions
  - RETRIEVAL_QUERY for search queries
- Matryoshka Representation Learning (MRL) truncation to 768 dimensions (or env EMBED_DIM)
- L2 unit vector normalization
- Batched document embedding (up to 100 items per request)
- Pixeltable query embedder UDF (embed_query)
"""

import os
import sys
import numpy as np
from typing import List, Optional
import pixeltable as pxt
from google import genai
from google.genai import types

# Register aliases for Pixeltable UDF serialization
sys.modules.setdefault("rag.embeddings", sys.modules[__name__])
sys.modules.setdefault("backend.rag.embeddings", sys.modules[__name__])

DIM = int(os.environ.get("EMBED_DIM", 768))
MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 100))

_api_key = os.environ.get("GEMINI_API_KEY")
_client: Optional[genai.Client] = genai.Client(api_key=_api_key) if _api_key else None


def _get_client() -> Optional[genai.Client]:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            _client = genai.Client(api_key=key)
    return _client


def _norm(v: np.ndarray | list[float]) -> np.ndarray:
    """Normalizes vector to L2 unit length. Required when truncating embedding dimensionality."""
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    return arr


def embed_docs(
    client: Optional[genai.Client],
    texts: List[str],
    batch: int = BATCH_SIZE,
    dim: int = DIM,
    model: str = MODEL,
) -> List[Optional[np.ndarray]]:
    """Generates task-typed (RETRIEVAL_DOCUMENT), L2-normalized embeddings in batches of up to `batch`.

    Returns a list of normalized np.ndarray vectors corresponding 1:1 with input texts.
    If a text is empty/whitespace or an embedding call fails, returns None for that item.
    """
    if not texts:
        return []

    if client is None:
        client = _get_client()

    if client is None:
        print("[embeddings] Warning: No Gemini API client available for doc embeddings.")
        return [None] * len(texts)

    results: List[Optional[np.ndarray]] = [None] * len(texts)

    # Process in chunks of `batch`
    for i in range(0, len(texts), batch):
        chunk_indices = []
        chunk_texts = []
        for idx in range(i, min(i + batch, len(texts))):
            t = texts[idx]
            if t and t.strip():
                chunk_indices.append(idx)
                chunk_texts.append(t.strip())

        if not chunk_texts:
            continue

        try:
            resp = client.models.embed_content(
                model=model,
                contents=chunk_texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dim,
                ),
            )
            if resp and resp.embeddings:
                for k, emb in enumerate(resp.embeddings):
                    if emb and getattr(emb, "values", None) is not None:
                        orig_idx = chunk_indices[k]
                        results[orig_idx] = _norm(emb.values)
        except Exception as e:
            print(f"[embeddings] Warning: Batch embedding failed for indices {i}..{i+batch}: {e}")
            # Individual items remain None (fail-open)

    return results


def _embed_query(
    text: str,
    client: Optional[genai.Client] = None,
    dim: int = DIM,
    model: str = MODEL,
) -> np.ndarray:
    """Core logic to generate a query-typed (RETRIEVAL_QUERY) normalized embedding vector."""
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    if client is None:
        client = _get_client()

    if client is None:
        return np.zeros(dim, dtype=np.float32)

    try:
        resp = client.models.embed_content(
            model=model,
            contents=text.strip(),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=dim,
            ),
        )
        if resp and resp.embeddings and getattr(resp.embeddings[0], "values", None) is not None:
            return _norm(resp.embeddings[0].values)
    except Exception as e:
        print(f"[embeddings] Warning: Query embedding failed for {text!r}: {e}")

    return np.zeros(dim, dtype=np.float32)


@pxt.udf
def embed_query(text: str) -> pxt.Array[(DIM,), pxt.Float]:
    """Pixeltable UDF for query-side embedding with task_type='RETRIEVAL_QUERY'."""
    return _embed_query(text)
