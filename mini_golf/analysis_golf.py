"""
Analyze mini-golf experiment results from results.tsv.
Adapted from autoresearch/analysis.ipynb for the mini-golf TSV schema.

Usage: python3 analysis_golf.py
"""
import sys
from pathlib import Path

import pandas as pd

TSV_PATH = Path(__file__).parent / "results.tsv"


def load_results() -> pd.DataFrame:
    if not TSV_PATH.exists():
        print(f"No results.tsv found at {TSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(TSV_PATH, sep="\t")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["artifact_bytes"] = pd.to_numeric(df["artifact_bytes"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()
    df["category"] = df["category"].str.strip().str.lower()
    df["escalate"] = df["escalate"].str.strip().str.lower()
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total experiments: {len(df)}")
    counts = df["status"].value_counts()
    for status in ["KEEP", "DISCARD", "CRASH"]:
        print(f"  {status}: {counts.get(status, 0)}")
    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_decided = n_keep + n_discard
    if n_decided > 0:
        print(f"  Keep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
    print()


def print_category_breakdown(df: pd.DataFrame) -> None:
    valid = df[df["status"] != "CRASH"].copy()
    if valid.empty:
        return
    print("Category breakdown (non-crash experiments):")
    print(f"  {'Category':<12} {'Total':>5} {'Kept':>5} {'Best BPB':>10}")
    print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*10}")
    for cat in sorted(valid["category"].unique()):
        cat_df = valid[valid["category"] == cat]
        kept = cat_df[cat_df["status"] == "KEEP"]
        best = cat_df["val_bpb"].min()
        print(f"  {cat:<12} {len(cat_df):>5} {len(kept):>5} {best:>10.6f}")
    print()


def print_kept_experiments(df: pd.DataFrame) -> None:
    kept = df[df["status"] == "KEEP"].copy()
    if kept.empty:
        print("No kept experiments yet.\n")
        return
    print(f"Kept experiments ({len(kept)} total):")
    for i, row in kept.iterrows():
        esc = " ** ESCALATE **" if row.get("escalate", "") == "yes" else ""
        print(f"  #{i:<3d}  bpb={row['val_bpb']:.6f}  [{row['category']}]  {row['description']}{esc}")
    print()


def print_improvement_deltas(df: pd.DataFrame) -> None:
    kept = df[df["status"] == "KEEP"].copy()
    if len(kept) < 2:
        return
    kept = kept.reset_index(drop=True)
    kept["prev_bpb"] = kept["val_bpb"].shift(1)
    kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]
    hits = kept.iloc[1:].copy()
    hits = hits.sort_values("delta", ascending=False)

    baseline_bpb = df.iloc[0]["val_bpb"]
    best_bpb = kept["val_bpb"].min()

    print(f"Baseline BPB:  {baseline_bpb:.6f}")
    print(f"Best BPB:      {best_bpb:.6f}")
    print(f"Improvement:   {baseline_bpb - best_bpb:.6f} ({(baseline_bpb - best_bpb) / baseline_bpb * 100:.2f}%)")
    print()
    print(f"  {'Rank':>4}  {'Delta':>9}  {'BPB':>10}  {'Cat':<10}  Description")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*30}")
    for rank, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"  {rank:4d}  {row['delta']:+.6f}  {row['val_bpb']:.6f}  {row['category']:<10}  {row['description']}")
    print(f"\n  {'':>4}  {hits['delta'].sum():+.6f}  {'':>10}  {'':>10}  TOTAL improvement")
    print()


def print_escalation_candidates(df: pd.DataFrame) -> None:
    esc = df[df.get("escalate", pd.Series(dtype=str)).str.strip().str.lower() == "yes"]
    if esc.empty:
        print("No experiments flagged for escalation yet.\n")
        return
    print(f"Escalation candidates ({len(esc)} flagged for H100 testing):")
    for _, row in esc.iterrows():
        print(f"  {row['commit']}  bpb={row['val_bpb']:.6f}  [{row['category']}]  {row['description']}")
    print()


def main() -> None:
    df = load_results()
    print("=" * 60)
    print("MINI-GOLF EXPERIMENT ANALYSIS")
    print("=" * 60)
    print()
    print_summary(df)
    print_category_breakdown(df)
    print_kept_experiments(df)
    print_improvement_deltas(df)
    print_escalation_candidates(df)


if __name__ == "__main__":
    main()
