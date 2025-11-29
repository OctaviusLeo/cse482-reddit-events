"""
Currently supports:
ROUGE-based evaluation of system summaries against a small human-written
"gold" set of reference summaries.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from rouge_score import rouge_scorer

from iris_reddit_events.config import SUMMARIES_DIR, DATA_DIR


def load_system_summaries(path: Path | None = None) -> pd.DataFrame:
    """
    Load system-generated event summaries.

    path : Path or None
        Optional explicit path; if None, use <SUMMARIES_DIR>/summaries.parquet.

    pd.DataFrame
        DataFrame with at least ['event_id', 'summary'] columns.
    """
    if path is None:
        path = SUMMARIES_DIR / "summaries.parquet"
    df = pd.read_parquet(path)
    if "event_id" not in df.columns or "summary" not in df.columns:
        raise ValueError("Expected columns ['event_id', 'summary'] in system summaries.")
    return df


def load_gold_references(path: Path | None = None) -> pd.DataFrame:
    """
    Load human-written reference summaries.

    path : Path or None
        CSV path with columns ['event_id', 'reference'].
        If None, default to <DATA_DIR>/gold_summaries.csv.

    pd.DataFrame
    """
    if path is None:
        path = DATA_DIR / "gold_summaries.csv"

    df = pd.read_csv(path)
    expected_cols = {"event_id", "reference"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"gold_summaries.csv must contain columns {expected_cols}")
    return df


def join_system_and_gold(
    sys_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join system and gold summaries on event_id.

    pd.DataFrame
        Columns: ['event_id', 'system', 'reference', ...]
    """
    merged = pd.merge(
        gold_df,
        sys_df[["event_id", "summary"]],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"summary": "system"})
    return merged


def compute_rouge(
    df: pd.DataFrame,
    metrics: Tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
) -> Dict[str, Dict[str, float]]:
    """
    Compute average ROUGE metrics over all rows.

    df : pd.DataFrame
        Must have columns ['reference', 'system'].
    metrics : tuple of str
        Any subset of ('rouge1', 'rouge2', 'rougeL', 'rougeLsum').

    dict
        {metric_name: {'precision': p, 'recall': r, 'fmeasure': f}}
        averaged over all rows.
    """
    scorer = rouge_scorer.RougeScorer(list(metrics), use_stemmer=True)
    agg = {m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for m in metrics}
    n = 0

    for _, row in df.iterrows():
        reference = str(row["reference"])
        system = str(row["system"])
        scores = scorer.score(reference, system)
        for m in metrics:
            agg[m]["precision"] += scores[m].precision
            agg[m]["recall"] += scores[m].recall
            agg[m]["fmeasure"] += scores[m].fmeasure
        n += 1

    if n == 0:
        return agg

    for m in metrics:
        for k in ("precision", "recall", "fmeasure"):
            agg[m][k] /= n

    return agg


def run_rouge_eval(
    gold_csv_path: str | Path | None = None,
    summaries_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Convenience function: load system+gold, compute ROUGE, print results.

    gold_csv_path : str or Path or None
        Path to gold_summaries.csv (defaults to DATA_DIR/gold_summaries.csv).
    summaries_path : str or Path or None
        Optional explicit path to system summaries parquet file.
        
    (df_eval, metrics)
        df_eval : pd.DataFrame with event_id, system, reference
        metrics : dict of averaged ROUGE stats
    """
    if gold_csv_path is None:
        gold_csv_path = DATA_DIR / "gold_summaries.csv"
    else:
        gold_csv_path = Path(gold_csv_path)

    if summaries_path is None:
        sys_df = load_system_summaries()
    else:
        sys_df = load_system_summaries(Path(summaries_path))

    gold_df = load_gold_references(gold_csv_path)
    df_eval = join_system_and_gold(sys_df, gold_df)
    metrics = compute_rouge(df_eval)

    print(f"Evaluated {len(df_eval)} event summaries with ROUGE.")
    for m, vals in metrics.items():
        print(
            f"{m}: "
            f"P={vals['precision']:.3f}  "
            f"R={vals['recall']:.3f}  "
            f"F1={vals['fmeasure']:.3f}"
        )

    return df_eval, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROUGE evaluation for event summaries.")
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold_summaries.csv (default: data/gold_summaries.csv)",
    )
    parser.add_argument(
        "--summaries",
        type=str,
        default=None,
        help="Optional path to system summaries parquet file.",
    )
    args = parser.parse_args()

    run_rouge_eval(args.gold, args.summaries)
