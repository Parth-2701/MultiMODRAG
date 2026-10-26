"""
Evaluation Runner for Video RAG Retrieval.
Usage:
    python eval/run_eval.py [--gold eval/gold.json] [--output eval/results.md] [--k 10]
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

# Add workspace root and backend to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from eval.metrics import hit_rank, recall_at_k, mrr
from eval.adapters import ADAPTERS

def parse_args():
    parser = argparse.ArgumentParser(description="Run Video RAG Evaluation Harness")
    parser.add_argument("--gold", type=str, default=str(ROOT / "eval" / "gold.json"), help="Path to gold.json")
    parser.add_argument("--output", type=str, default=str(ROOT / "eval" / "results.md"), help="Path to output markdown report")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument("--adapters", type=str, default=None, help="Comma-separated adapter names to run")
    return parser.parse_args()

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    d = k - f
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * d

def run_evaluation(gold_path: str, chosen_adapters: list[str], k: int = 10):
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", [])
    records = []

    print(f"\n[eval] Running evaluation on {len(videos)} videos with {len(chosen_adapters)} adapters (top-k={k})...\n")

    for adapter_name in chosen_adapters:
        adapter_fn = ADAPTERS[adapter_name]
        print(f"[eval] Evaluating Adapter: [{adapter_name}]")

        for video in videos:
            title = video["title"]
            questions = video.get("questions", [])

            for q in questions:
                qid = q["id"]
                qtype = q.get("type", "needle")
                query = q["question"]
                gold_spans = q.get("gold_spans", [])

                t0 = time.perf_counter()
                try:
                    retrieved = adapter_fn(title, query, k=k)
                except Exception as e:
                    print(f"   [warn] Error running {adapter_name} on {title} ({qid}): {e}")
                    retrieved = []
                lat_ms = (time.perf_counter() - t0) * 1000.0

                rank = hit_rank(retrieved, gold_spans)
                r1 = recall_at_k(rank, 1)
                r5 = recall_at_k(rank, 5)
                r10 = recall_at_k(rank, 10)
                mrr_score = mrr(rank)

                records.append({
                    "adapter": adapter_name,
                    "video": title,
                    "id": qid,
                    "type": qtype,
                    "rank": rank,
                    "r1": r1,
                    "r5": r5,
                    "r10": r10,
                    "mrr": mrr_score,
                    "latency_ms": lat_ms,
                })

    return records

def build_markdown_report(records: list[dict], output_path: str, k: int):
    # Group by adapter overall
    by_adapter = defaultdict(list)
    # Group by (adapter, question_type)
    by_adapter_type = defaultdict(list)

    for r in records:
        by_adapter[r["adapter"]].append(r)
        by_adapter_type[(r["adapter"], r["type"])].append(r)

    lines = []
    lines.append("# Video RAG Retrieval Evaluation Results")
    lines.append(f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append(f"Evaluated with `k={k}` retrieval limit across benchmark question sets.\n")

    # 1. Overall Summary Table
    lines.append("## Overall Performance by Strategy\n")
    lines.append("| Strategy / Adapter | Recall@1 | Recall@5 | Recall@10 | MRR | Latency p50 (ms) | Latency p95 (ms) | Total Queries |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    console_summary = []

    for adapter, rows in by_adapter.items():
        n = len(rows)
        r1_avg = sum(r["r1"] for r in rows) / n if n else 0
        r5_avg = sum(r["r5"] for r in rows) / n if n else 0
        r10_avg = sum(r["r10"] for r in rows) / n if n else 0
        mrr_avg = sum(r["mrr"] for r in rows) / n if n else 0
        lats = [r["latency_ms"] for r in rows]
        p50 = percentile(lats, 50)
        p95 = percentile(lats, 95)

        line = f"| **{adapter}** | {r1_avg:.3f} | {r5_avg:.3f} | {r10_avg:.3f} | {mrr_avg:.3f} | {p50:.1f} | {p95:.1f} | {n} |"
        lines.append(line)
        console_summary.append(line)

    lines.append("\n---\n")

    # 2. Performance Grouped by Question Type (Needle vs Temporal vs Summary)
    lines.append("## Breakdown by Question Type\n")
    lines.append("| Strategy | Question Type | Recall@1 | Recall@5 | Recall@10 | MRR | Count |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for (adapter, qtype), rows in sorted(by_adapter_type.items()):
        n = len(rows)
        r1_avg = sum(r["r1"] for r in rows) / n if n else 0
        r5_avg = sum(r["r5"] for r in rows) / n if n else 0
        r10_avg = sum(r["r10"] for r in rows) / n if n else 0
        mrr_avg = sum(r["mrr"] for r in rows) / n if n else 0
        lines.append(f"| {adapter} | `{qtype}` | {r1_avg:.3f} | {r5_avg:.3f} | {r10_avg:.3f} | {mrr_avg:.3f} | {n} |")

    report_text = "\n".join(lines) + "\n"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[eval] Detailed Markdown report saved to: {output_path}\n")
    except Exception as e:
        print(f"\n[warn] Could not write report to {output_path} ({e}).")

    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 90)
    for c in console_summary:
        print(c)
    print("=" * 90)

def main():
    args = parse_args()
    if args.adapters:
        chosen = [a.strip() for a in args.adapters.split(",") if a.strip() in ADAPTERS]
    else:
        chosen = list(ADAPTERS.keys())

    records = run_evaluation(args.gold, chosen, k=args.k)
    build_markdown_report(records, args.output, k=args.k)

if __name__ == "__main__":
    main()
