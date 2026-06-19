"""
Cache correctness evaluation: compares cached answers vs fresh LLM answers.

DATA STATUS: [MISSING_MANUAL_CACHE_REVIEW]
Semantic equivalence between a cached answer and a fresh answer cannot be
judged automatically without a gold-label or human review. This script:
  1. Sends the same query twice (first to populate cache, second to hit it)
  2. Records whether a cache hit occurred and how long each path took
  3. Computes cosine similarity between the two answer embeddings as a proxy
  4. Saves results to evaluation/results/cache_correctness.csv for manual review

Manual reviewers should fill in the 'manually_equivalent' column (1/0/skip).
Until that column is filled, the file is labeled [MISSING_MANUAL_CACHE_REVIEW].

Usage:
    python3 scripts/eval_cache_correctness.py
    python3 scripts/eval_cache_correctness.py --api http://localhost:8000 --n 10
    python3 scripts/eval_cache_correctness.py --summarize  # after manual review

Outputs:
    evaluation/results/cache_correctness.csv
    evaluation/results/cache_correctness_summary.csv  (after manual review)
"""

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "cache_correctness.csv"
SUMMARY_CSV = RESULTS_DIR / "cache_correctness_summary.csv"

DEFAULT_API = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Representative test queries — varied across agents
TEST_QUERIES = [
    {"question": "CTL-01 makinesi şu an çalışıyor mu?", "agent_collection": "digital_twin"},
    {"question": "DST-01 sensör sıcaklığı nedir?", "agent_collection": "digital_twin"},
    {"question": "Servo besleme ünitesinin teknik özellikleri nelerdir?", "agent_collection": "product_catalog"},
    {"question": "CTL serisinde kaç model var?", "agent_collection": "product_catalog"},
    {"question": "Decoiler arızası nasıl giderilmiş?", "agent_collection": "service_maintenance"},
    {"question": "Enkoder hatası için servis kaydı var mı?", "agent_collection": "service_maintenance"},
    {"question": "OEE değeri nedir?", "agent_collection": "digital_twin"},
    {"question": "Yedek parça stok durumu nedir?", "agent_collection": "spare_parts"},
    {"question": "CE belgesi nasıl alınır?", "agent_collection": "quality_docs"},
    {"question": "Son çeyrekte kaç arıza kaydı var?", "agent_collection": "management_report"},
]


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def _embed_text(client: httpx.AsyncClient, text: str) -> list:
    resp = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": -1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def _ask(client: httpx.AsyncClient, api_base: str, question: str) -> tuple[str, float, bool]:
    """Returns (answer_text, latency_seconds, is_cache_hit)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api_base}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            },
            timeout=120,
        )
        latency = time.monotonic() - t0
        if resp.status_code != 200:
            return f"[ERROR {resp.status_code}]", latency, False
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        # Heuristic: cache hits are typically < 0.5s and much faster than LLM
        is_cache_hit = latency < 0.5
        return answer, round(latency, 3), is_cache_hit
    except Exception as e:
        latency = time.monotonic() - t0
        return f"[EXCEPTION: {e}]", round(latency, 3), False


async def run_eval(api_base: str, n: int) -> None:
    queries = TEST_QUERIES[:n]
    print(f"Cache correctness eval: {len(queries)} queries")
    print(f"API: {api_base}\n")
    print("Step 1: Invalidate cache and get fresh answers...")

    rows = []
    async with httpx.AsyncClient() as http:
        # First pass: get fresh answers (will populate cache)
        fresh_answers = []
        for i, q in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {q['question'][:55]}")
            answer, latency, _ = await _ask(http, api_base, q["question"])
            fresh_answers.append({"answer": answer, "latency": latency})
            await asyncio.sleep(0.5)  # brief pause to avoid rate limiting

        print("\nStep 2: Re-send same queries (should hit cache)...")
        for i, (q, fresh) in enumerate(zip(queries, fresh_answers)):
            print(f"  [{i+1}/{len(queries)}] {q['question'][:55]}")
            cached_answer, cached_latency, is_hit = await _ask(http, api_base, q["question"])

            # Embed both answers for similarity
            fresh_emb = await _embed_text(http, fresh["answer"][:1000])
            cached_emb = await _embed_text(http, cached_answer[:1000])
            similarity = _cosine(fresh_emb, cached_emb)

            print(f"       fresh={fresh['latency']:.2f}s  cached={cached_latency:.2f}s  "
                  f"sim={similarity:.3f}  hit={'YES' if is_hit else 'NO'}")

            rows.append({
                "question": q["question"],
                "agent_collection": q["agent_collection"],
                "fresh_latency_s": fresh["latency"],
                "cached_latency_s": cached_latency,
                "speedup_x": round(fresh["latency"] / cached_latency, 2) if cached_latency > 0 else 0,
                "cache_hit_heuristic": is_hit,
                "answer_cosine_similarity": round(similarity, 4),
                "fresh_answer": fresh["answer"][:300],
                "cached_answer": cached_answer[:300],
                "manually_equivalent": "[MISSING_MANUAL_CACHE_REVIEW]",
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    avg_sim = sum(r["answer_cosine_similarity"] for r in rows) / len(rows)
    avg_speedup = sum(r["speedup_x"] for r in rows) / len(rows)
    cache_hits = sum(1 for r in rows if r["cache_hit_heuristic"])
    print(f"\nCache hit rate (heuristic): {cache_hits}/{len(rows)}")
    print(f"Average answer similarity:  {avg_sim:.4f}")
    print(f"Average speedup:            {avg_speedup:.1f}x")
    print(f"\nSaved: {OUT_CSV}")
    print("\n[MISSING_MANUAL_CACHE_REVIEW] — manually fill 'manually_equivalent' column,")
    print("then run: python3 scripts/eval_cache_correctness.py --summarize")


def summarize() -> None:
    """Compute final correctness stats after manual review."""
    if not OUT_CSV.exists():
        print(f"No results file found: {OUT_CSV}")
        raise SystemExit(1)

    with open(OUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reviewed = [r for r in rows if r["manually_equivalent"] not in ("", "[MISSING_MANUAL_CACHE_REVIEW]", "skip")]
    if not reviewed:
        print("[MISSING_MANUAL_CACHE_REVIEW] — no rows have been manually reviewed yet.")
        raise SystemExit(1)

    equiv = sum(1 for r in reviewed if r["manually_equivalent"] == "1")
    total = len(reviewed)
    accuracy = equiv / total * 100
    avg_speedup = sum(float(r["speedup_x"]) for r in reviewed) / total

    print(f"Manual review results: {equiv}/{total} ({accuracy:.1f}%) answers equivalent")
    print(f"Average speedup: {avg_speedup:.1f}x")

    summary = [
        ["metric", "value"],
        ["queries_reviewed", total],
        ["equivalent_answers", equiv],
        ["correctness_pct", round(accuracy, 2)],
        ["avg_speedup_x", round(avg_speedup, 2)],
        ["missing_manual_review", len(rows) - total],
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary)
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache correctness evaluation")
    parser.add_argument("--api", default=os.environ.get("BACKEND_API_URL", DEFAULT_API))
    parser.add_argument("--n", type=int, default=10, help="Number of queries to test (max 10)")
    parser.add_argument("--summarize", action="store_true",
                        help="Compute stats from manually-reviewed CSV (no API calls)")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        asyncio.run(run_eval(args.api, min(args.n, len(TEST_QUERIES))))
