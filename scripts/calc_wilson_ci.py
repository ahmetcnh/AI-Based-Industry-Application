"""
Wilson score interval calculator for routing accuracy claims.

Usage:
    python scripts/calc_wilson_ci.py
    python scripts/calc_wilson_ci.py --correct 43 --total 44 --z 1.96
    python scripts/calc_wilson_ci.py --correct 43 --total 44 --z 1.645  # 90% CI

Outputs a CSV row to evaluation/results/routing_wilson_ci.csv and prints the interval.
Formula: Wilson (1927) — handles small samples and extreme proportions correctly.
"""

import argparse
import csv
import math
from pathlib import Path

OUT = Path(__file__).parent.parent / "evaluation" / "results" / "routing_wilson_ci.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def wilson_ci(correct: int, total: int, z: float = 1.96):
    """Return (lower, center, upper) Wilson score interval."""
    if total == 0:
        raise ValueError("total must be > 0")
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, center - margin), center, min(1.0, center + margin)


def main():
    parser = argparse.ArgumentParser(description="Wilson CI for routing accuracy")
    parser.add_argument("--correct", type=int, default=43)
    parser.add_argument("--total", type=int, default=44)
    parser.add_argument("--z", type=float, default=1.96, help="z-score for CI (1.96 for 95 pct, 1.645 for 90 pct)")
    args = parser.parse_args()

    lo, center, hi = wilson_ci(args.correct, args.total, args.z)
    p_hat = args.correct / args.total

    confidence = {1.645: 90, 1.96: 95, 2.576: 99}.get(args.z, round(args.z, 3))

    print(f"Correct:    {args.correct}/{args.total}")
    print(f"p̂ (MLE):   {p_hat:.4f}  ({p_hat*100:.2f}%)")
    print(f"CI ({confidence}%):  [{lo:.4f}, {hi:.4f}]  ({lo*100:.2f}% – {hi*100:.2f}%)")
    print(f"Center:     {center:.4f}  ({center*100:.2f}%)")
    print()
    print(f"Thesis claim: {args.correct}/{args.total} = {p_hat*100:.1f}%  "
          f"[{lo*100:.1f}%–{hi*100:.1f}%] at {confidence}% CI")

    rows = [
        ["correct", "total", "p_hat", "ci_lower", "ci_center", "ci_upper",
         "z_score", "confidence_pct", "claim"],
        [args.correct, args.total, round(p_hat, 6), round(lo, 6), round(center, 6),
         round(hi, 6), args.z, confidence,
         f"{args.correct}/{args.total}={p_hat*100:.1f}% [{lo*100:.1f}%–{hi*100:.1f}%] at {confidence}% CI"],
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
