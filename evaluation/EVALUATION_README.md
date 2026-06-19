# Evaluation Pipeline — Coiltech RAG Platform

Reproducible evaluation suite for the graduation thesis claims.
All scripts live in `scripts/`, all outputs in `evaluation/results/`, charts in `eval/charts/`.

---

## Prerequisites

```bash
# Docker stack must be running
docker compose up -d

# Ollama must be running with nomic-embed-text
ollama pull nomic-embed-text

# Python deps (from project root)
pip3 install asyncpg httpx matplotlib numpy
```

---

## Scripts

### Task 6 — Wilson Confidence Interval (fully computable, no stack needed)
```bash
python3 scripts/calc_wilson_ci.py
# Output: evaluation/results/routing_wilson_ci.csv
```
- Computes 95% Wilson CI for 43/44 routing accuracy
- Result: 97.7% [88.2%–99.6%] at 95% CI

---

### Task 2 — Router Ablation on 44-Question Benchmark
```bash
python3 scripts/eval_router_ambiguous.py
# Output: evaluation/results/ablation_tier_metrics.csv
#         evaluation/results/ablation_predictions.csv
#         eval/charts/8_router_ablation.png
```
- Compares: Tier 1 (embedding only) vs Tier 2 (+keyword boost) vs Tier 3 (+LLM)
- Uses hand-curated 44-question benchmark (`eval/routing_questions.json`)
- Tier 3 requires backend API running at `http://localhost:8000`

---

### Task 3 — Generated Corpus Routing (543 questions)

> **WARNING:** Corpus was template-generated and NOT independently validated.
> All results from this script must be labeled `NOT_INDEPENDENTLY_VALIDATED`.

```bash
python3 scripts/eval_generated_corpus.py
# Output: evaluation/results/generated_corpus_metrics.csv
#         evaluation/results/generated_corpus_predictions.csv
#         eval/charts/7_generated_corpus.png
```

---

### Task 1 — Retrieval Strategy Comparison

> **[MISSING_GOLD_RETRIEVAL_LABELS]** — requires manual annotation first.

```bash
# 1. Fill in evaluation/retrieval_benchmark_template.csv
#    Rename to evaluation/retrieval_benchmark.csv
#    Add chunk IDs from:
#      SELECT id FROM chunks WHERE content ILIKE '%keyword%' LIMIT 5;

# 2. Then run:
python3 scripts/eval_retrieval.py --k 5 --alpha 0.3
# Output: evaluation/results/retrieval_metrics.csv
#         evaluation/results/retrieval_per_query.csv
#         eval/charts/9_retrieval_comparison.png
```

---

### Task 4 — Cache Correctness

> **[MISSING_MANUAL_CACHE_REVIEW]** — requires manual review after running.

```bash
# Step 1: Generate answers and cache hits
python3 scripts/eval_cache_correctness.py --n 10
# Output: evaluation/results/cache_correctness.csv

# Step 2: Manually fill 'manually_equivalent' column (1/0/skip)

# Step 3: Compute final stats
python3 scripts/eval_cache_correctness.py --summarize
# Output: evaluation/results/cache_correctness_summary.csv
```

---

### Task 5 — Baseline Comparison
```bash
# Keyword-only baseline (no stack needed)
python3 scripts/eval_baselines.py --mode keyword

# Full comparison (stack required)
python3 scripts/eval_baselines.py --mode all
# Output: evaluation/results/baseline_comparison.csv
#         evaluation/results/baseline_metrics.csv
#         eval/charts/10_baseline_comparison.png
```

---

## Output Files

| File | Task | Status |
|------|------|--------|
| `evaluation/results/routing_wilson_ci.csv` | Task 6 | Ready |
| `evaluation/results/ablation_tier_metrics.csv` | Task 2 | Requires stack |
| `evaluation/results/ablation_predictions.csv` | Task 2 | Requires stack |
| `evaluation/results/generated_corpus_metrics.csv` | Task 3 | Requires stack; NOT_INDEPENDENTLY_VALIDATED |
| `evaluation/results/generated_corpus_predictions.csv` | Task 3 | Requires stack; NOT_INDEPENDENTLY_VALIDATED |
| `evaluation/results/retrieval_metrics.csv` | Task 1 | MISSING_GOLD_RETRIEVAL_LABELS |
| `evaluation/results/retrieval_per_query.csv` | Task 1 | MISSING_GOLD_RETRIEVAL_LABELS |
| `evaluation/results/cache_correctness.csv` | Task 4 | MISSING_MANUAL_CACHE_REVIEW |
| `evaluation/results/cache_correctness_summary.csv` | Task 4 | After manual review |
| `evaluation/results/baseline_comparison.csv` | Task 5 | Requires stack |
| `evaluation/results/baseline_metrics.csv` | Task 5 | Requires stack |

---

## Thesis Claims Audit

See `evaluation/results/claims_audit.md` for the mapping between thesis claims
and the scripts/data that support or refute each claim.
