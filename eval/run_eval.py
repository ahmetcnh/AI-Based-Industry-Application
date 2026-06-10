"""
Evaluation script for the Coiltech RAG Platform.

Usage:
    python eval/run_eval.py [--model MODEL] [--base-url URL]

Requires the live stack to be running:
    docker compose up -d backend

Then run from the repo root:
    python eval/run_eval.py
    python eval/run_eval.py --model coiltech-rag --base-url http://localhost:8000

Exit code 0 if pass rate >= 60%, 1 otherwise.
"""

import argparse
import asyncio
import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import httpx

QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def _normalize(text: str) -> str:
    """Lowercase and normalize Turkish characters for comparison."""
    replacements = {
        "ı": "i", "İ": "i", "ş": "s", "Ş": "s",
        "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
        "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    }
    text = text.lower()
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return unicodedata.normalize("NFKD", text)


def _keywords_found(answer: str, keywords: List[str]) -> List[str]:
    norm_answer = _normalize(answer)
    missing = []
    for kw in keywords:
        if _normalize(kw) not in norm_answer:
            missing.append(kw)
    return missing


async def evaluate(base_url: str, model: str) -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for q in questions:
            t0 = time.time()
            try:
                resp = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": q["question"]}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
            except Exception as e:
                answer = f"ERROR: {e}"
            duration = time.time() - t0
            missing = _keywords_found(answer, q["keywords"])
            results.append({
                "id": q["id"],
                "question": q["question"][:60],
                "passed": len(missing) == 0,
                "missing": missing,
                "duration": duration,
                "answer_snippet": answer[:80],
            })

    # Print table
    header = f"{'ID':<4} {'Pass':<5} {'Duration':>9}  {'Question':<62}  {'Missing Keywords'}"
    print(header)
    print("-" * 120)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        missing_str = ", ".join(r["missing"]) if r["missing"] else "-"
        print(f"{r['id']:<4} {status:<5} {r['duration']:>8.1f}s  {r['question']:<62}  {missing_str}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    durations = [r["duration"] for r in results]
    avg_d = sum(durations) / len(durations) if durations else 0
    sorted_d = sorted(durations)
    p95_d = sorted_d[int(len(sorted_d) * 0.95)] if sorted_d else 0

    print("-" * 120)
    print(f"Pass rate: {passed}/{total} ({passed/total*100:.1f}%)  avg={avg_d:.1f}s  p95={p95_d:.1f}s")

    if passed / total < 0.6:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="coiltech-rag", help="Model ID (default: coiltech-rag)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()
    asyncio.run(evaluate(args.base_url, args.model))
