"""
Baseline comparison: multi-agent router vs single-agent vs keyword-only.

Compares three system configurations on the 44-question hand-curated benchmark:

  A. Keyword-only:  simple substring/keyword match (no embedding, no LLM)
  B. Single-agent:  all questions go to a single generic RAG agent
  C. Multi-agent:   production router (Tier 1+2+3) dispatches to correct agent

Note: configurations B and C require the backend stack to be running.
Configuration A is purely local and needs only the questions file.

Usage:
    python3 scripts/eval_baselines.py --mode all  # requires running stack
    python3 scripts/eval_baselines.py --mode keyword  # local only
    python3 scripts/eval_baselines.py --dsn postgresql://... --api http://localhost:8000

Outputs:
    evaluation/results/baseline_comparison.csv
    evaluation/results/baseline_metrics.csv
    eval/charts/10_baseline_comparison.png
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time
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

_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


def _norm(text: str) -> str:
    return text.translate(_TR_MAP).lower()


# ── Keyword-only baseline ──────────────────────────────────────────────────
_KEYWORD_RULES = [
    ("Dijital İkiz",     ["dijital ikiz", "makine durumu", "ctl-0", "ack-0", "dst-0", "oee",
                          "sensor", "sensör", "alarm", "canli", "canlı", "sicaklik", "basinc"]),
    ("Servis Kayıtları", ["servis", "ariza kaydi", "arıza kaydı", "bakim kaydi", "nasil giderilmis",
                          "nasıl giderilmiş", "daha once", "daha önce"]),
    ("Ürün Kataloğu",    ["ctl", "decoiler", "destacker", "teknik ozellik", "datasheet",
                          "spesifikasyon", "hangi makineler", "modelleri"]),
    ("Yedek Parça",      ["stok", "yedek parca", "yedek parça", "tedarik", "envanter",
                          "rulman", "conta", "filtre", "siparis", "sipariş"]),
    ("Kurulum",          ["kurulum", "montaj", "devreye alma", "komisyon", "hizalama"]),
    ("Satış",            ["fiyat teklifi", "rfq", "proforma", "odeme", "ödeme", "satis sozlesme"]),
    ("Kalite",           ["ce belgesi", "iso", "uygunluk beyanı", "kalite dokuman",
                          "sertifika", "belgelendirme"]),
    ("Yönetim Raporu",   ["istatistik", "ozet rapor", "aylik ariza", "kpi", "toplam kac",
                          "trend analiz", "yonetici"]),
    ("Özel Projeler",    ["milestone", "proje takvim", "gantt", "wbs", "kilometre tasi"]),
]


def keyword_route(question: str) -> str:
    n = _norm(question)
    for agent, keywords in _KEYWORD_RULES:
        if any(kw in n for kw in keywords):
            return agent
    return "Dijital İkiz"  # fallback


# ── Multi-agent routing via production router ──────────────────────────────
async def _multi_agent_route(client: httpx.AsyncClient, api_base: str, question: str) -> tuple[str, float]:
    """Route via the production system. Returns (predicted_agent, latency_s)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api_base}/v1/route",
            json={"message": question, "tenant_id": 1, "user_id": 1},
            timeout=30,
        )
        latency = time.monotonic() - t0
        if resp.status_code == 200:
            data = resp.json()
            agent = data.get("agent") or data.get("routed_to") or "unknown"
            return agent, round(latency, 3)
        return f"[HTTP {resp.status_code}]", round(latency, 3)
    except Exception as e:
        return f"[ERROR: {e}]", round(time.monotonic() - t0, 3)


# ── Single-agent: route everything to one generic agent ────────────────────
SINGLE_AGENT_NAME = "Servis Kayıtları"  # largest agent by document count


async def main(dsn: str, api_base: str, mode: str) -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    print(f"Questions: {len(questions)}  mode={mode}\n")

    results = []
    counts = {"keyword": 0, "single_agent": 0, "multi_agent": 0}
    latencies = {"multi_agent": []}

    async with httpx.AsyncClient() as http:
        for i, q in enumerate(questions):
            text = q["question"]
            expected = q["expected_agent"]
            row = {"question": text, "expected": expected}

            # A: keyword-only
            pred_kw = keyword_route(text)
            ok_kw = pred_kw == expected
            counts["keyword"] += ok_kw
            row["pred_keyword"] = pred_kw
            row["ok_keyword"] = ok_kw

            # B: single-agent
            pred_single = SINGLE_AGENT_NAME
            ok_single = pred_single == expected
            counts["single_agent"] += ok_single
            row["pred_single"] = pred_single
            row["ok_single"] = ok_single

            # C: multi-agent
            if mode in ("all", "multi"):
                pred_multi, lat = await _multi_agent_route(http, api_base, text)
                ok_multi = pred_multi == expected
                counts["multi_agent"] += ok_multi
                latencies["multi_agent"].append(lat)
                row["pred_multi"] = pred_multi
                row["ok_multi"] = ok_multi
                row["multi_latency_s"] = lat
                mark = "✓" if ok_multi else "✗"
                print(f"  {mark} {text[:55]:<55}  kw={'✓' if ok_kw else '✗'}  "
                      f"single={'✓' if ok_single else '✗'}  multi={pred_multi[:20]}")
            else:
                row["pred_multi"] = "N/A"
                row["ok_multi"] = "N/A"
                row["multi_latency_s"] = "N/A"
                print(f"  kw={'✓' if ok_kw else '✗'}  single={'✓' if ok_single else '✗'}  "
                      f"multi=SKIPPED  [{text[:40]}]")

            results.append(row)

    total = len(questions)
    print(f"\n── Baseline comparison (n={total}) ──")
    configs = [
        ("Keyword-only",  "keyword",      "(simple substring match, no embedding)"),
        ("Single-agent",  "single_agent", f"(all → {SINGLE_AGENT_NAME})"),
        ("Multi-agent",   "multi_agent",  "(production 3-tier router)"),
    ]
    for label, key, note in configs:
        if mode == "keyword" and key == "multi_agent":
            print(f"  {label:<20}  SKIPPED  {note}")
            continue
        acc = counts[key] / total * 100
        print(f"  {label:<20}  {counts[key]}/{total}  ({acc:.1f}%)  {note}")

    # Save predictions
    pred_path = RESULTS_DIR / "baseline_comparison.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nPredictions saved: {pred_path}")

    # Save metrics
    metrics_path = RESULTS_DIR / "baseline_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "description", "correct", "total", "accuracy_pct",
                    "avg_route_latency_s", "note"])
        for label, key, note in configs:
            if mode == "keyword" and key == "multi_agent":
                w.writerow([label, note, "N/A", total, "N/A", "N/A", "skipped"])
                continue
            acc = counts[key] / total * 100
            avg_lat = (sum(latencies["multi_agent"]) / len(latencies["multi_agent"])
                       if key == "multi_agent" and latencies["multi_agent"] else "N/A")
            w.writerow([label, note, counts[key], total, round(acc, 2), avg_lat, ""])
    print(f"Metrics saved:     {metrics_path}")

    _chart(counts, total, mode)


def _chart(counts: dict, total: int, mode: str) -> None:
    labels = ["Keyword\nonly", "Single\nagent", "Multi-agent\n(production)"]
    keys = ["keyword", "single_agent", "multi_agent"]
    accs = []
    for key in keys:
        if mode == "keyword" and key == "multi_agent":
            accs.append(0)
        else:
            accs.append(counts[key] / total * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#EF9A9A", "#FFCC80", "#1565C0"]
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, width=0.5, color=colors, edgecolor="white", linewidth=1.2)

    for bar, acc, key in zip(bars, accs, keys):
        if mode == "keyword" and key == "multi_agent":
            ax.text(bar.get_x() + bar.get_width() / 2, 2, "N/A",
                    ha="center", fontsize=10, color="#888")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{counts[key]}/{total}\n{acc:.1f}%",
                    ha="center", fontsize=10.5, fontweight="bold", color="#1A237E")

    ax.axhline(85, color="#BDBDBD", linewidth=1.2, linestyle="--")
    ax.text(2.4, 86, "85% target", fontsize=9, color="#757575", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=11, color="#444")
    ax.set_title(f"Baseline Comparison  (n={total})",
                 fontsize=12, fontweight="bold", color="#111", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCC")
    ax.tick_params(colors="#333")

    plt.tight_layout()
    out = OUT_DIR / "10_baseline_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved:       {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline comparison: keyword / single-agent / multi-agent")
    parser.add_argument("--dsn", default=os.environ.get("ROUTER_EVAL_DSN", DEFAULT_DSN))
    parser.add_argument("--api", default=os.environ.get("BACKEND_API_URL", DEFAULT_API))
    parser.add_argument("--mode", choices=["all", "keyword", "multi"], default="all",
                        help="'keyword' skips API calls, 'all' runs all three configs")
    args = parser.parse_args()
    asyncio.run(main(args.dsn, args.api, args.mode))
