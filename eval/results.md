# Video RAG Retrieval Evaluation Results

*Generated: 2026-10-26 17:33:26*

Evaluated with `k=10` retrieval limit across benchmark question sets.

## Overall Performance by Strategy

| Strategy / Adapter | Recall@1 | Recall@5 | Recall@10 | MRR | Latency p50 (ms) | Latency p95 (ms) | Total Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_dense_proxy** | 0.600 | 0.867 | 0.867 | 0.722 | 701.7 | 16530.0 | 15 |
| **dense_only** | 0.933 | 1.000 | 1.000 | 0.967 | 676.5 | 715.4 | 15 |
| **bm25_only** | 0.800 | 1.000 | 1.000 | 0.900 | 81.0 | 194.5 | 15 |
| **hybrid_rrf_spans** | 1.000 | 1.000 | 1.000 | 1.000 | 782.7 | 821.5 | 15 |

---

## Breakdown by Question Type

| Strategy | Question Type | Recall@1 | Recall@5 | Recall@10 | MRR | Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| baseline_dense_proxy | `needle` | 0.333 | 0.778 | 0.778 | 0.537 | 9 |
| baseline_dense_proxy | `summary` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| baseline_dense_proxy | `temporal` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| bm25_only | `needle` | 0.889 | 1.000 | 1.000 | 0.944 | 9 |
| bm25_only | `summary` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| bm25_only | `temporal` | 0.333 | 1.000 | 1.000 | 0.667 | 3 |
| dense_only | `needle` | 0.889 | 1.000 | 1.000 | 0.944 | 9 |
| dense_only | `summary` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| dense_only | `temporal` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| hybrid_rrf_spans | `needle` | 1.000 | 1.000 | 1.000 | 1.000 | 9 |
| hybrid_rrf_spans | `summary` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| hybrid_rrf_spans | `temporal` | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
