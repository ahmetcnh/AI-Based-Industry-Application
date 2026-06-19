# Thesis Claims Audit

Maps each quantitative claim in the thesis to its evidence source and verification status.

---

## Claim 1: Routing Accuracy 97.7% (43/44)

**Thesis location:** Chapter 4, Section 4.1 (Semantic Router), Table 4.x

**Evidence:**
- Script: `eval/eval_routing.py`
- Benchmark: `eval/routing_questions.json` (44 questions, hand-curated)
- Wilson 95% CI: **88.2% – 99.6%** → see `evaluation/results/routing_wilson_ci.csv`

**Status:** VERIFIED
- 43/44 correct on the hand-curated benchmark
- Wilson CI accounts for small sample size (n=44)
- The single error is documented in eval_routing.py output

**Caveats:**
- n=44 is small; CI is wide (88%–99.6%)
- Questions were written by the thesis author, not an independent evaluator
- Template-generated 543-question corpus (NOT_INDEPENDENTLY_VALIDATED) is a separate claim

---

## Claim 2: 543-Question Corpus Accuracy

**Thesis location:** Chapter 4, if referenced

**Evidence:**
- Script: `scripts/eval_generated_corpus.py`
- Corpus: `eval/routing_questions_500.json` (543 questions)
- Output: `evaluation/results/generated_corpus_metrics.csv`

**Status:** NOT_INDEPENDENTLY_VALIDATED
- Corpus was generated automatically from agent name/description templates
- Questions and expected_agent labels were not manually verified
- Results indicate system behavior on template-generated questions, not real user queries
- **Do not cite as an independent benchmark**

---

## Claim 3: Three-Tier Router Improvement over Embedding-Only

**Thesis location:** Chapter 4, Section 4.1

**Evidence:**
- Script: `scripts/eval_router_ambiguous.py`
- Output: `evaluation/results/ablation_tier_metrics.csv`

**Status:** REQUIRES_STACK (run with Docker compose up)
- Tier 1 (embedding only) vs Tier 2 (+keyword boost) vs Tier 3 (+LLM) on 44-question benchmark
- Expected: Tier 2 and Tier 3 improve on Tier 1, especially on ambiguous questions
- Ambiguity threshold: 0.07 cosine gap

---

## Claim 4: Hybrid Retrieval Outperforms BM25 and Dense Alone

**Thesis location:** Chapter 4, Section 4.2 (Hybrid Retrieval), alpha=0.3

**Evidence:**
- Script: `scripts/eval_retrieval.py`
- Template: `evaluation/retrieval_benchmark_template.csv`

**Status:** [MISSING_GOLD_RETRIEVAL_LABELS]
- No chunk-level relevance annotations exist
- Must fill `evaluation/retrieval_benchmark.csv` with at least 30 annotated queries
- SQL to find relevant chunks: `SELECT id FROM chunks WHERE content ILIKE '%keyword%'`

**Interim claim:** Hybrid retrieval is theoretically motivated by the complementary coverage
of BM25 (exact term match) and dense embeddings (semantic similarity). The alpha=0.3
weight was chosen empirically; a formal retrieval eval is deferred to future work.

---

## Claim 5: Semantic Cache Reduces Latency

**Thesis location:** Chapter 4, Section 4.7 (Semantic Answer Cache)

**Evidence:**
- Script: `scripts/eval_cache_correctness.py`
- Output: `evaluation/results/cache_correctness.csv` (after running)

**Status:** [MISSING_MANUAL_CACHE_REVIEW]
- Cache hit latency vs fresh answer latency is measured automatically
- Answer equivalence requires manual review (semantic similarity alone is insufficient)
- Run `python3 scripts/eval_cache_correctness.py`, then fill 'manually_equivalent' column

**Interim claim:** Cache hits are expected to be ~100× faster (LLM generation ~2–5s
vs cache lookup ~0.05–0.1s). Answer correctness depends on threshold (θ=0.95).

---

## Claim 6: Multi-Agent Outperforms Single-Agent and Keyword Baseline

**Thesis location:** Chapter 5 (Evaluation / Discussion)

**Evidence:**
- Script: `scripts/eval_baselines.py`
- Output: `evaluation/results/baseline_metrics.csv`

**Status:** REQUIRES_STACK
- Keyword-only baseline runs locally (no stack needed): `--mode keyword`
- Single-agent baseline assumes all questions → one agent (trivial worst case for specialized queries)
- Multi-agent production router: `--mode all` requires Docker stack

---

## Summary Table

| Claim | Script | Status |
|-------|--------|--------|
| Routing 97.7% on 44q | `eval/eval_routing.py` + `scripts/calc_wilson_ci.py` | VERIFIED |
| 543q corpus accuracy | `scripts/eval_generated_corpus.py` | NOT_INDEPENDENTLY_VALIDATED |
| 3-tier ablation | `scripts/eval_router_ambiguous.py` | REQUIRES_STACK |
| Hybrid retrieval best | `scripts/eval_retrieval.py` | MISSING_GOLD_RETRIEVAL_LABELS |
| Cache latency reduction | `scripts/eval_cache_correctness.py` | MISSING_MANUAL_CACHE_REVIEW |
| Multi-agent vs baselines | `scripts/eval_baselines.py` | REQUIRES_STACK |
