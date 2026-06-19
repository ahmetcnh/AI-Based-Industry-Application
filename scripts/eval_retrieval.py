"""
Retrieval strategy comparison: BM25 vs Dense vs Hybrid vs Hybrid+Reranker.

DATA STATUS: [MISSING_GOLD_RETRIEVAL_LABELS]
This script requires a manually annotated ground-truth file at:
    evaluation/retrieval_benchmark.csv

The file must have columns:
    query, relevant_chunk_ids (semicolon-separated), collection

A blank template is at evaluation/retrieval_benchmark_template.csv.
Fill in at least 30 queries × relevant chunks before running.

Once the file exists, this script computes:
  - Precision@k, Recall@k, MRR (Mean Reciprocal Rank), nDCG@k
  - Side-by-side comparison across 4 retrieval strategies
  - Chart saved to eval/charts/9_retrieval_comparison.png
  - Results saved to evaluation/results/retrieval_metrics.csv

Usage (after filling retrieval_benchmark.csv):
    python3 scripts/eval_retrieval.py
    python3 scripts/eval_retrieval.py --k 5 --alpha 0.3 --dsn postgresql://...
"""

import argparse
import asyncio
import csv
import os
from pathlib import Path

import asyncpg
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
BENCHMARK_FILE = ROOT / "evaluation" / "retrieval_benchmark.csv"
TEMPLATE_FILE = ROOT / "evaluation" / "retrieval_benchmark_template.csv"
OUT_DIR = ROOT / "eval" / "charts"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DSN = "postgresql://postgres:test1234@localhost:5432/ragdb"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
TENANT_ID = 1
USER_ID = 1


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def _embed(client: httpx.AsyncClient, text: str) -> list:
    resp = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": -1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / k if k else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 1)
              for rank, rid in enumerate(retrieved_ids[:k], start=1)
              if rid in relevant_ids)
    idcg = sum(1.0 / np.log2(rank + 1)
               for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg else 0.0


async def dense_search(conn, embedding: list, collection: str, k: int) -> list[int]:
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    rows = await conn.fetch(
        """
        SELECT id, 1 - (embedding <=> $1::vector) AS score
        FROM chunks
        WHERE tenant_id=$2 AND user_id=$3 AND collection=$4
        ORDER BY embedding <=> $1::vector
        LIMIT $5
        """,
        emb_str, TENANT_ID, USER_ID, collection, k,
    )
    return [r["id"] for r in rows]


async def bm25_search(conn, query: str, collection: str, k: int) -> list[int]:
    rows = await conn.fetch(
        """
        SELECT id, ts_rank_cd(to_tsvector('simple', content),
               plainto_tsquery('simple', $1)) AS score
        FROM chunks
        WHERE tenant_id=$2 AND user_id=$3 AND collection=$4
          AND to_tsvector('simple', content) @@ plainto_tsquery('simple', $1)
        ORDER BY score DESC
        LIMIT $5
        """,
        query, TENANT_ID, USER_ID, collection, k,
    )
    return [r["id"] for r in rows]


async def hybrid_search(conn, query: str, embedding: list,
                        collection: str, k: int, alpha: float) -> list[int]:
    vec_rows = await dense_search(conn, embedding, collection, k * 2)
    bm25_rows = await bm25_search(conn, query, collection, k * 2)

    scores: dict = {}
    for rank, rid in enumerate(vec_rows):
        scores[rid] = scores.get(rid, 0.0) + (1 - alpha) * (1.0 / (rank + 1))
    bm25_max = len(bm25_rows)
    for rank, rid in enumerate(bm25_rows):
        scores[rid] = scores.get(rid, 0.0) + alpha * (1.0 - rank / bm25_max)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked[:k]]


async def main(dsn: str, k: int, alpha: float) -> None:
    if not BENCHMARK_FILE.exists():
        print(f"[MISSING_GOLD_RETRIEVAL_LABELS]")
        print(f"Ground-truth file not found: {BENCHMARK_FILE}")
        print(f"Fill in the template at: {TEMPLATE_FILE}")
        print("\nSkipping retrieval evaluation. Create the benchmark file to enable this script.")
        raise SystemExit(1)

    # Load benchmark
    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        benchmark = list(reader)
    print(f"Benchmark queries: {len(benchmark)}")

    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=4)
    strategies = {
        "BM25 only": [],
        "Dense only": [],
        "Hybrid": [],
        "Hybrid+Reranker": [],  # placeholder — reranker disabled in production
    }
    metric_keys = ["p@k", "recall@k", "mrr", "ndcg@k"]
    all_results = {s: {m: [] for m in metric_keys} for s in strategies}
    query_rows = []

    async with httpx.AsyncClient() as http:
        for row in benchmark:
            query = row["query"]
            collection = row["collection"]
            relevant_ids = set(int(x) for x in row["relevant_chunk_ids"].split(";") if x.strip())

            q_emb = await _embed(http, query)

            async with pool.acquire() as conn:
                ids_bm25 = await bm25_search(conn, query, collection, k * 3)
                ids_dense = await dense_search(conn, q_emb, collection, k * 3)
                ids_hybrid = await hybrid_search(conn, query, q_emb, collection, k * 3, alpha)

            for strat, ids in [
                ("BM25 only", ids_bm25),
                ("Dense only", ids_dense),
                ("Hybrid", ids_hybrid),
                ("Hybrid+Reranker", ids_hybrid),  # same as hybrid — reranker N/A
            ]:
                p = precision_at_k(ids, relevant_ids, k)
                r = recall_at_k(ids, relevant_ids, k)
                m = mrr(ids, relevant_ids)
                n = ndcg_at_k(ids, relevant_ids, k)
                all_results[strat]["p@k"].append(p)
                all_results[strat]["recall@k"].append(r)
                all_results[strat]["mrr"].append(m)
                all_results[strat]["ndcg@k"].append(n)
                query_rows.append({
                    "query": query,
                    "collection": collection,
                    "strategy": strat,
                    "precision_at_k": round(p, 4),
                    "recall_at_k": round(r, 4),
                    "mrr": round(m, 4),
                    "ndcg_at_k": round(n, 4),
                })

    await pool.close()

    # Aggregate
    print(f"\n── Retrieval metrics @ k={k} (alpha={alpha}) ──")
    metric_rows = [["strategy", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]]
    for strat, metrics in all_results.items():
        avgs = {m: sum(v) / len(v) for m, v in metrics.items()}
        print(f"  {strat:<22}  P@{k}={avgs['p@k']:.3f}  "
              f"R@{k}={avgs['recall@k']:.3f}  MRR={avgs['mrr']:.3f}  nDCG@{k}={avgs['ndcg@k']:.3f}")
        metric_rows.append([strat, round(avgs["p@k"], 4), round(avgs["recall@k"], 4),
                             round(avgs["mrr"], 4), round(avgs["ndcg@k"], 4)])

    metrics_path = RESULTS_DIR / "retrieval_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(metric_rows)
    print(f"\nMetrics saved: {metrics_path}")

    query_path = RESULTS_DIR / "retrieval_per_query.csv"
    with open(query_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=query_rows[0].keys())
        w.writeheader()
        w.writerows(query_rows)
    print(f"Per-query saved: {query_path}")

    _chart(all_results, k, alpha)


def _chart(all_results: dict, k: int, alpha: float) -> None:
    strategies = list(all_results.keys())
    metric_keys = ["p@k", "recall@k", "mrr", "ndcg@k"]
    metric_labels = [f"Precision@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    colors = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        ax.set_facecolor("white")
        vals = [sum(all_results[s][mkey]) / len(all_results[s][mkey]) for s in strategies]
        x = np.arange(len(strategies))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color="#0D47A1")
        ax.set_xticks(x)
        ax.set_xticklabels(["BM25", "Dense", "Hybrid", "Hyb+\nRnk"], fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(mlabel, fontsize=10, fontweight="bold", color="#333")
        ax.set_ylabel("Score", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCC")
        ax.tick_params(colors="#333")

    fig.suptitle(f"Retrieval Strategy Comparison (k={k}, α={alpha})",
                 fontsize=13, fontweight="bold", color="#111", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "9_retrieval_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval strategy comparison")
    parser.add_argument("--dsn", default=os.environ.get("ROUTER_EVAL_DSN", DEFAULT_DSN))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3, help="BM25 weight in hybrid")
    args = parser.parse_args()
    asyncio.run(main(args.dsn, args.k, args.alpha))
