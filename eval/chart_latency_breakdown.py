"""
Latency breakdown chart — stacked horizontal bars showing time spent
in each pipeline stage per agent type.

Stages: Router Embed | Vector Search | LLM Generation | DB Query | Cache Lookup
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Measured / estimated stage times (seconds) ───────────────────────────────
#
# SQL Agent (cache miss):  embed 0.35 + sql_llm 1.1 + db 0.02 + summary_llm 1.0  ≈ 2.5s
# SQL Agent (cache hit):   cache_lookup 0.05                                      ≈ 0.05s
# RAG Agent (Gemini):      embed 0.35 + vec_search 0.08 + llm 8.5                 ≈ 9.0s
# RAG Agent (Ollama):      embed 0.35 + vec_search 0.08 + llm 137.0               ≈ 137.5s

CONFIGS = [
    {
        "label": "Ollama + RAG\n(Local CPU)",
        "stages": {
            "Router Embed\n(nomic)":  0.35,
            "Vector Search\n(pgvector)": 0.08,
            "LLM Generation\n(Ollama gemma3)": 137.0,
            "DB Query\n(SQL)": 0.0,
            "Cache Lookup": 0.0,
        },
    },
    {
        "label": "Gemini + Router\n+ Full RAG",
        "stages": {
            "Router Embed\n(nomic)":  0.35,
            "Vector Search\n(pgvector)": 0.08,
            "LLM Generation\n(Gemini 2.5 Flash)": 8.5,
            "DB Query\n(SQL)": 0.0,
            "Cache Lookup": 0.0,
        },
    },
    {
        "label": "SQL Agent\n(Gemini, no cache)",
        "stages": {
            "Router Embed\n(nomic)":  0.35,
            "Vector Search\n(pgvector)": 0.0,
            "LLM Generation\n(Gemini 2.5 Flash)": 2.1,
            "DB Query\n(SQL)": 0.02,
            "Cache Lookup": 0.0,
        },
    },
    {
        "label": "SQL Agent\n(Cache Hit)",
        "stages": {
            "Router Embed\n(nomic)":  0.0,
            "Vector Search\n(pgvector)": 0.0,
            "LLM Generation\n(Gemini 2.5 Flash)": 0.0,
            "DB Query\n(SQL)": 0.0,
            "Cache Lookup": 0.05,
        },
    },
]

STAGE_COLORS = {
    "Router Embed\n(nomic)":              "#AED6F1",   # light blue
    "Vector Search\n(pgvector)":          "#85C1E9",   # medium blue
    "LLM Generation\n(Gemini 2.5 Flash)": "#2E86C1",   # dark blue
    "LLM Generation\n(Ollama gemma3)":    "#1A5276",   # very dark blue
    "DB Query\n(SQL)":                    "#7FB3D3",   # steel blue
    "Cache Lookup":                       "#D6E4F0",   # near-white blue
}

STAGE_LABELS_SHORT = {
    "Router Embed\n(nomic)":              "Router Embed",
    "Vector Search\n(pgvector)":          "Vector Search",
    "LLM Generation\n(Gemini 2.5 Flash)": "LLM (Gemini)",
    "LLM Generation\n(Ollama gemma3)":    "LLM (Ollama)",
    "DB Query\n(SQL)":                    "DB Query",
    "Cache Lookup":                       "Cache Lookup",
}


def build_breakdown():
    all_stages = list(STAGE_COLORS.keys())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(CONFIGS))
    bar_h = 0.52

    for i, cfg in enumerate(CONFIGS):
        left = 0.0
        for stage in all_stages:
            val = cfg["stages"].get(stage, 0.0)
            if val <= 0:
                continue
            color = STAGE_COLORS[stage]
            ax.barh(i, val, left=left, height=bar_h,
                    color=color, edgecolor="white", linewidth=0.8)
            # Inline label for segments wide enough to read
            if val > 0.3:
                label_x = left + val / 2
                ax.text(label_x, i, f"{val:.2f}s",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white",
                        path_effects=[
                            __import__("matplotlib.patheffects", fromlist=["withStroke"])
                            .withStroke(linewidth=1.5, foreground="#333333")
                        ])
            left += val

        # Total label at end of bar
        total = sum(v for v in cfg["stages"].values() if v > 0)
        ax.text(left + total * 0.015 + 0.3, i, f"Total: {total:.2f}s",
                va="center", ha="left", fontsize=9.5, fontweight="bold",
                color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c["label"] for c in CONFIGS], fontsize=10.5)
    ax.invert_yaxis()

    # Use log scale only if range spans >2 orders of magnitude
    max_t = max(
        sum(v for v in c["stages"].values() if v > 0)
        for c in CONFIGS
    )
    ax.set_xscale("log")
    ax.set_xlim(0.01, max_t * 2.5)
    ax.set_xlabel("Time (seconds, log scale)", fontsize=11, color="#444444")

    ax.tick_params(colors="#333333", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.3, color="#CCCCCC")

    ax.set_title(
        "Pipeline Latency Breakdown  —  Time per Stage by Agent Type",
        fontsize=13, fontweight="bold", color="#111111", pad=12,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=STAGE_COLORS[s], label=STAGE_LABELS_SHORT[s],
                       edgecolor="#AAAAAA")
        for s in all_stages
        if any(c["stages"].get(s, 0) > 0 for c in CONFIGS)
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right",
              facecolor="white", edgecolor="#CCCCCC", framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "5_latency_breakdown.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✔ Saved: {out}")


if __name__ == "__main__":
    print("── Building latency breakdown chart ──")
    build_breakdown()
