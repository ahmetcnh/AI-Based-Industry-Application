"""
Routing accuracy on the 543-question template-generated corpus.

WARNING: The corpus in routing_questions_500.json was generated automatically
from agent name/description templates and has NOT been manually validated.
Accuracy numbers from this script are indicative only and must be labeled
NOT_INDEPENDENTLY_VALIDATED in any publication.

Usage:
    python3 scripts/eval_generated_corpus.py
    python3 scripts/eval_generated_corpus.py --dsn postgresql://... --no-boost

Outputs:
    evaluation/results/generated_corpus_predictions.csv
    evaluation/results/generated_corpus_metrics.csv
    eval/charts/7_generated_corpus.png
"""

import argparse
import asyncio
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import asyncpg
import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
QUESTIONS_FILE = ROOT / "eval" / "routing_questions_500.json"
OUT_DIR = ROOT / "eval" / "charts"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DSN = "postgresql://postgres:test1234@localhost:5432/ragdb"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")

# Keyword boosts — copied from eval/eval_routing.py to stay in sync
_MACHINE_NAMES = [
    "ctl", "decoiler", "destacker", "blank hat", "pres transfer",
    "servo besleme", "loop surucü", "kompakt surucü", "servo surucü",
    "acici", "açici", "dogrultma", "doğrultma",
]
_SPEC_WORDS = [
    "teknik ozellik", "teknik özelli", "datasheet", "spesifikasyon",
    "konfigurasyonlar", "voltaji", "voltajı", "ailesinde", "modelleri",
    "hangi makineler", "kapasitesi", "maksimum kapasite",
]
_KEYWORD_BOOSTS = [
    {
        "agent": "Yönetim Raporu",
        "any": [
            "istatistik", "ozet rapor", "aylik ariza ozeti", "toplam kac",
            "kac servis", "yonetim icin", "kpi", "kategori dagilim",
            "aylik ozet", "sayisal ozet", "son ceyrekte", "bu yil kac",
            "ortalama ariza", "ariza sayisi", "makine basina", "yonetici ozeti",
            "trend analiz", "ceyreklik", "istatistiksel dokum", "uzun vadeli ariza",
            "ariza frekan", "ariza sikligi", "ariza orani", "bakim maliyeti",
            "servis istatistik", "makine guvenirlili",
        ],
        "boost": 0.12,
    },
    {
        "agent": "Özel Projeler",
        "any": [
            "milestone", "kilometre tasi", "proje takvim", "gantt",
            "proje zaman", "teslim takvim", "wbs", "proje tutanak",
            "proje ilerles", "proje kapsam", "proje fizibilite",
            "proje muhendis", "proje kapani", "proje deliverable",
            "proje charter", "proje tamamlan", "proje ekib",
            "proje butce", "proje bütce", "proje paydas",
            "proje risk", "proje aktivite", "proje basa", "proje ilerleme",
        ],
        "boost": 0.12,
    },
    {
        "agent": "Ürün Kataloğu",
        "any": _MACHINE_NAMES,
        "also": _SPEC_WORDS,
        "boost": 0.15,
    },
]


def _norm(text: str) -> str:
    return text.translate(_TR_MAP).lower()


def _apply_boosts(scored: list, question_norm: str) -> list:
    result = list(scored)
    for rule in _KEYWORD_BOOSTS:
        target = rule["agent"]
        idx = next((i for i, (_, n) in enumerate(result) if n == target), None)
        if idx is None:
            continue
        if not any(kw in question_norm for kw in rule["any"]):
            continue
        also = rule.get("also")
        if also and not any(kw in question_norm for kw in also):
            continue
        s, name = result[idx]
        result[idx] = (s + rule["boost"], name)
    return result


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def _embed(client: httpx.AsyncClient, text: str):
    resp = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": -1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def load_agents(dsn: str):
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT name, embedding FROM agents "
            "WHERE tenant_id=1 AND user_id=1 AND embedding IS NOT NULL"
        )
    finally:
        await conn.close()
    agents = []
    for row in rows:
        raw = row["embedding"]
        emb = [float(x) for x in raw.strip("[]").split(",")] if isinstance(raw, str) else list(raw)
        agents.append((row["name"], emb))
    return agents


async def main(dsn: str, use_boost: bool = True) -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    agents = await load_agents(dsn)
    print(f"Agents: {[n for n, _ in agents]}")
    print(f"Questions: {len(questions)}  (template-generated corpus — NOT_INDEPENDENTLY_VALIDATED)")
    print(f"Boost: {'enabled' if use_boost else 'disabled'}\n")

    per_agent: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    correct = 0

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions):
            q_emb = await _embed(client, q["question"])
            scored = [(_cosine(q_emb, emb), name) for name, emb in agents]
            if use_boost:
                scored = _apply_boosts(scored, _norm(q["question"]))
            scored.sort(reverse=True)
            top_score, predicted = scored[0]
            expected = q["expected_agent"]
            ok = predicted == expected
            correct += ok
            per_agent[expected]["total"] += 1
            per_agent[expected]["correct"] += ok
            predictions.append({
                "idx": i,
                "question": q["question"],
                "expected": expected,
                "predicted": predicted,
                "top_score": round(top_score, 4),
                "correct": ok,
                "validation_status": "NOT_INDEPENDENTLY_VALIDATED",
            })
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(questions)} ({correct}/{i+1} correct so far)")

    total = len(questions)
    accuracy = correct / total * 100

    print(f"\nTotal accuracy: {correct}/{total}  ({accuracy:.1f}%)  — NOT_INDEPENDENTLY_VALIDATED")

    # Per-agent metrics
    metrics_rows = [["agent", "correct", "total", "accuracy_pct", "validation_status"]]
    for agent, st in sorted(per_agent.items()):
        pct = st["correct"] / st["total"] * 100
        metrics_rows.append([agent, st["correct"], st["total"],
                              round(pct, 2), "NOT_INDEPENDENTLY_VALIDATED"])
        print(f"  {agent:<28} {st['correct']}/{st['total']}  ({pct:.1f}%)")

    # Add summary row
    metrics_rows.append(["TOTAL", correct, total, round(accuracy, 2), "NOT_INDEPENDENTLY_VALIDATED"])

    # Save predictions
    pred_path = RESULTS_DIR / "generated_corpus_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=predictions[0].keys())
        w.writeheader()
        w.writerows(predictions)
    print(f"\nPredictions saved: {pred_path}")

    # Save metrics
    metrics_path = RESULTS_DIR / "generated_corpus_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(metrics_rows)
    print(f"Metrics saved:     {metrics_path}")

    _chart(per_agent, accuracy)


def _chart(per_agent: dict, accuracy: float) -> None:
    names = sorted(per_agent.keys())
    pcts = [per_agent[n]["correct"] / per_agent[n]["total"] * 100 for n in names]
    totals = [per_agent[n]["total"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(names))
    bars = ax.bar(x, pcts, width=0.6, color="#1976D2", edgecolor="white", linewidth=1.2)
    for bar, p, t in zip(bars, pcts, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{p:.0f}%\nn={t}", ha="center", fontsize=8.5, fontweight="bold", color="#1565C0")

    ax.axhline(85, color="#90CAF9", linewidth=1.2, linestyle="--")
    ax.text(len(names) - 0.5, 86.5, "target 85%", fontsize=9, color="#64B5F6", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ve ", " &\n").replace(" ", "\n", 1)
                        for n in names], fontsize=8.5)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=11, color="#444")
    ax.set_title(
        f"Router Accuracy — Generated Corpus (n=543)  |  Overall: {accuracy:.1f}%  "
        f"[NOT INDEPENDENTLY VALIDATED]",
        fontsize=11, fontweight="bold", color="#111", pad=10,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCC")
    ax.tick_params(colors="#333")

    plt.tight_layout()
    out = OUT_DIR / "7_generated_corpus.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved:       {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval routing on 543-question generated corpus")
    parser.add_argument("--dsn", default=os.environ.get("ROUTER_EVAL_DSN", DEFAULT_DSN))
    parser.add_argument("--no-boost", dest="use_boost", action="store_false",
                        help="disable keyword boosts (pure embedding baseline)")
    args = parser.parse_args()
    asyncio.run(main(args.dsn, args.use_boost))
