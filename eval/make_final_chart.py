"""Final presentation — Response Time by Agent Type chart.
Style: matches existing eval/charts/* (white bg, blue palette, clean sans-serif).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette (same as existing charts) ──────────────────────────────────
C_DARK   = "#1565C0"   # dark blue  — SQL agent / labels
C_MED    = "#1E88E5"   # medium blue — RAG agents
C_LIGHT  = "#90CAF9"   # light blue  — Digital Twin
C_GREEN  = "#2E7D32"   # green       — cache hit
C_BG     = "#FFFFFF"
C_GRID   = "#DDDDDD"
C_AXIS   = "#333333"

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ── Data ──────────────────────────────────────────────────────────────────────
labels = [
    "Digital Twin\n(real-time data)",
    "Cache Hit\n(any agent)",
    "SQL Agent\n(no cache)",
    "Spare Parts\n(RAG)",
    "Installation\n(RAG)",
    "Quality\n(RAG)",
    "Sales\n(RAG)",
    "Product Catalog\n(RAG)",
]
values  = [0.09, 0.05, 2.47,  8.7, 9.1, 9.3, 8.9, 9.4]
colors  = [C_LIGHT, C_GREEN, C_DARK, C_MED, C_MED, C_MED, C_MED, C_MED]

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8, zorder=3)

# ── Labels on bars ────────────────────────────────────────────────────────────
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val}s",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color=C_DARK if val > 1 else C_GREEN if val == 0.05 else C_AXIS,
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5, color=C_AXIS)
ax.set_ylabel("Response Time  (seconds)", fontsize=12, color=C_AXIS, labelpad=10)
ax.set_ylim(0, 11.5)
ax.yaxis.set_tick_params(labelcolor=C_AXIS, labelsize=10)
ax.xaxis.set_tick_params(length=0)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# ── Annotation: cache speedup ─────────────────────────────────────────────────
ax.annotate(
    "180× faster\nthan RAG",
    xy=(1, 0.05), xytext=(1, 4.5),
    fontsize=9.5, color=C_GREEN, fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4),
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_MED,   label="RAG Agent (hybrid search + LLM answer)"),
    mpatches.Patch(color=C_DARK,  label="SQL Agent (NL→SQL + DB query + LLM summary)"),
    mpatches.Patch(color=C_LIGHT, label="Digital Twin (deterministic real-time data)"),
    mpatches.Patch(color=C_GREEN, label="Semantic Cache Hit (pgvector ≥ 0.95 similarity)"),
]
ax.legend(
    handles=legend_handles,
    loc="upper left", bbox_to_anchor=(0.01, 0.99),
    fontsize=9.5,
    frameon=True, framealpha=0.95, edgecolor=C_GRID,
)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    "Coiltech Multi-Agent RAG  —  Response Time by Agent Type\n"
    "Gemini 2.5 Flash backend  ·  pgvector hybrid search  ·  ms-marco reranker",
    fontsize=13, fontweight="bold", color=C_AXIS, pad=14,
)

plt.tight_layout()
out = "/home/ubuntu/bitirme2/eval/charts/final_response_time.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out}")
