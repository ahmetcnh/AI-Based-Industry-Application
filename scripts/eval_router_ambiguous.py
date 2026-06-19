"""
Three-tier router ablation on the 44-question hand-curated benchmark.

Compares three configurations:
  Tier 1: embedding-only (cosine similarity, no boosts)
  Tier 2: embedding + keyword boosts
  Tier 3: embedding + boosts + LLM disambiguation (full production router)

The LLM disambiguation is called only when the top-2 cosine gap is < AMBIGUITY_THRESHOLD (0.07).
Requires:
  - Docker stack running (postgres, backend on port 8000)
  - Ollama running locally with nomic-embed-text

Usage:
    python3 scripts/eval_router_ambiguous.py
    python3 scripts/eval_router_ambiguous.py --dsn postgresql://... --api http://localhost:8000

Outputs:
    evaluation/results/ablation_tier_metrics.csv
    evaluation/results/ablation_predictions.csv
    eval/charts/8_router_ablation.png
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
QUESTIONS_FILE = ROOT / "eval" / "routing_questions.json"
OUT_DIR = ROOT / "eval" / "charts"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DSN = "postgresql://postgres:test1234@localhost:5432/ragdb"
DEFAULT_API = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
AMBIGUITY_THRESHOLD = 0.07  # must match backend/router_agent.py

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")

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


async def _embed(client: httpx.AsyncClient, text: str) -> list:
    resp = await client.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": -1},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def _llm_route(client: httpx.AsyncClient, api_base: str,
                     question: str, candidates: list[str]) -> str | None:
    """Call the backend's routing debug endpoint or fallback to /v1/chat/completions."""
    try:
        # Try the internal route-only endpoint first
        resp = await client.post(
            f"{api_base}/v1/route",
            json={"message": question, "tenant_id": 1, "user_id": 1},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("agent") or data.get("routed_to")
    except Exception:
        pass

    # Fallback: ask the LLM directly to pick an agent
    agent_list = "\n".join(f"- {a}" for a in candidates)
    prompt = (
        f"Kullanıcı sorusu: {question}\n\n"
        f"Mevcut ajanlar:\n{agent_list}\n\n"
        "Bu soruyu yanıtlamak için en uygun ajanı seçin. "
        "Sadece ajan adını yaz, başka hiçbir şey yazma."
    )
    try:
        resp = await client.post(
            f"{api_base}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Find best match
            for candidate in candidates:
                if candidate.lower() in content.lower():
                    return candidate
    except Exception:
        pass
    return None


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


async def main(dsn: str, api_base: str) -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    agents = await load_agents(dsn)
    agent_names = [n for n, _ in agents]
    print(f"Agents ({len(agents)}): {agent_names}")
    print(f"Questions: {len(questions)}\n")

    results = []
    tier_correct = {"t1_embed_only": 0, "t2_boost": 0, "t3_llm": 0}
    ambiguous_count = 0

    async with httpx.AsyncClient() as http:
        for q in questions:
            text = q["question"]
            expected = q["expected_agent"]
            q_emb = await _embed(http, text)
            q_norm = _norm(text)

            # Tier 1: embedding only
            scored_t1 = sorted([(_cosine(q_emb, emb), name) for name, emb in agents], reverse=True)
            pred_t1 = scored_t1[0][1]

            # Tier 2: + keyword boosts
            scored_t2 = sorted(_apply_boosts(list(scored_t1), q_norm), reverse=True)
            pred_t2 = scored_t2[0][1]

            # Check ambiguity on tier-2 scores
            gap = scored_t2[0][0] - scored_t2[1][0] if len(scored_t2) > 1 else 1.0
            is_ambiguous = gap < AMBIGUITY_THRESHOLD
            ambiguous_count += is_ambiguous

            # Tier 3: + LLM for ambiguous cases
            if is_ambiguous:
                top_candidates = [n for _, n in scored_t2[:3]]
                llm_result = await _llm_route(http, api_base, text, top_candidates)
                pred_t3 = llm_result if llm_result else pred_t2
            else:
                pred_t3 = pred_t2

            ok1 = pred_t1 == expected
            ok2 = pred_t2 == expected
            ok3 = pred_t3 == expected
            tier_correct["t1_embed_only"] += ok1
            tier_correct["t2_boost"] += ok2
            tier_correct["t3_llm"] += ok3

            mark = lambda ok: "✓" if ok else "✗"
            print(f"  {mark(ok3)} {text[:55]:<55}  gap={gap:.3f}{'  [AMB]' if is_ambiguous else ''}")
            if not ok3:
                print(f"      expected={expected}  t1={pred_t1}  t2={pred_t2}  t3={pred_t3}")

            results.append({
                "question": text,
                "expected": expected,
                "pred_t1_embed_only": pred_t1,
                "pred_t2_boost": pred_t2,
                "pred_t3_llm": pred_t3,
                "cosine_gap": round(gap, 4),
                "is_ambiguous": is_ambiguous,
                "ok_t1": ok1,
                "ok_t2": ok2,
                "ok_t3": ok3,
            })

    total = len(questions)
    print(f"\nAmbiguous cases (gap < {AMBIGUITY_THRESHOLD}): {ambiguous_count}/{total}")
    print("\n── Tier accuracy comparison ──")
    labels = {
        "t1_embed_only": "Tier 1: Embedding only",
        "t2_boost":      "Tier 2: + Keyword boosts",
        "t3_llm":        "Tier 3: + LLM (production)",
    }
    tier_acc = {}
    for key, label in labels.items():
        acc = tier_correct[key] / total * 100
        tier_acc[key] = acc
        print(f"  {label:<32}  {tier_correct[key]}/{total}  ({acc:.1f}%)")

    # Save predictions
    pred_path = RESULTS_DIR / "ablation_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nPredictions saved: {pred_path}")

    # Save tier metrics
    metrics_path = RESULTS_DIR / "ablation_tier_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tier", "label", "correct", "total", "accuracy_pct",
                    "ambiguous_cases", "ambiguity_threshold"])
        for key, label in labels.items():
            w.writerow([key, label, tier_correct[key], total,
                        round(tier_acc[key], 2), ambiguous_count, AMBIGUITY_THRESHOLD])
    print(f"Tier metrics saved: {metrics_path}")

    _chart(tier_correct, tier_acc, total, labels, ambiguous_count)


def _chart(tier_correct, tier_acc, total, labels, ambiguous_count):
    keys = list(labels.keys())
    short_labels = ["Tier 1\nEmbed only", "Tier 2\n+Keyword boost", "Tier 3\n+LLM (prod)"]
    accs = [tier_acc[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#90CAF9", "#42A5F5", "#1565C0"]
    x = np.arange(len(keys))
    bars = ax.bar(x, accs, width=0.5, color=colors, edgecolor="white", linewidth=1.2)
    for bar, acc, key in zip(bars, accs, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{tier_correct[key]}/{total}\n{acc:.1f}%",
                ha="center", fontsize=10.5, fontweight="bold", color="#1A237E")

    ax.axhline(85, color="#BDBDBD", linewidth=1.2, linestyle="--")
    ax.text(2.4, 86, "85% target", fontsize=9, color="#757575", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=11, color="#444")
    ax.set_title(
        f"Routing Ablation: 3-Tier Architecture  (n={total}, "
        f"ambiguous={ambiguous_count})",
        fontsize=12, fontweight="bold", color="#111", pad=10,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCC")
    ax.tick_params(colors="#333")

    plt.tight_layout()
    out = OUT_DIR / "8_router_ablation.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved:        {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Routing ablation: embed-only vs boost vs LLM")
    parser.add_argument("--dsn", default=os.environ.get("ROUTER_EVAL_DSN", DEFAULT_DSN))
    parser.add_argument("--api", default=os.environ.get("BACKEND_API_URL", DEFAULT_API),
                        help="Backend API base URL")
    args = parser.parse_args()
    asyncio.run(main(args.dsn, args.api))
