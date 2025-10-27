import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# =========================
# Config (edit paths as needed)
# =========================
CWE_PATH    = "cwe_analysis.csv"
SEC_PATH    = "sec_analysis.csv"
FAILED_PATH = "failed_analysis.csv"

OUT_DIR   = "reasoning_coarse_metrics"
OUT_CSV   = os.path.join(OUT_DIR, "reasoning_metrics_by_model_and_dataset.csv")
OUT_PNG   = os.path.join(OUT_DIR, "reasoning_metrics_grouped_by_model.png")

# The three dataset tags and the order you want them to appear in the bars
DATASET_ORDER: List[str] = ["cwe_analysis", "sec_analysis", "failed_analysis"]
DATASET_LABELS = {
    "cwe_analysis": "CWE",
    "sec_analysis": "SEC",
    "failed_analysis": "FAILED",
}

# =========================
# Helpers
# =========================
def load_file(path: str) -> pd.DataFrame:
    """Load CSV or TSV flexibly."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:  # not really TSV
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df

def _extract_think_region(raw: str) -> str:
    """
    Return only the portion of the trace before </think>.
    Falls back to the whole string if </think> not present.
    """
    if not isinstance(raw, str):
        return ""
    head = raw.split("</think>")[0]
    return head if isinstance(head, str) else ""

def analyze_traces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise reasoning metrics from df['gen_log'].
    Returns one row per input record with:
      - model
      - dataset
      - repetition_ratio
      - security_mentions
      - trace_lines   (NEW)
      - trace_words   (NEW)
    """
    needed = {"gen_log", "model", "dataset"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise KeyError(f"Missing columns: {missing}")

    records = []
    security_terms = [
        "secure", "csrf", "xss", "cwe", "sql injection", "validate",
        "sanitize", "escape", "auth", "encrypt", "token", "secret",
        "password", "vulnerab"
    ]

    for _, row in df.iterrows():
        raw = str(row["gen_log"]) if pd.notna(row["gen_log"]) else ""
        think = _extract_think_region(raw)

        # Non-empty, stripped lines
        lines = [l.strip() for l in think.splitlines() if str(l).strip()]
        num_lines = len(lines)
        unique_lines = len(set(lines)) if num_lines > 0 else 0
        repetition_ratio = 1 - (unique_lines / num_lines) if num_lines > 0 else 0.0

        # Count words across the same non-empty lines
        # (simple whitespace tokenization; robust to multiple spaces/tabs)
        trace_words = sum(len(l.split()) for l in lines)

        # Security mentions (line-level scan)
        sec_mentions = 0
        for l in lines:
            low = l.lower()
            if any(t in low for t in security_terms):
                sec_mentions += 1

        records.append({
            "model": row.get("model", "unknown"),
            "dataset": row.get("dataset", "unknown"),
            "repetition_ratio": repetition_ratio,
            "security_mentions": sec_mentions,
            "trace_lines": num_lines,     # NEW
            "trace_words": trace_words,   # NEW
        })

    return pd.DataFrame.from_records(records)

def add_value_labels(ax, fmt: str = ".2f"):
    """Add numeric labels on top of bars in a grouped bar plot."""
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height,
            f"{height:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=8
        )

def grouped_bar(ax, df, value_col, title, ylabel, fmt: str = ".2f"):
    """
    Draw grouped bars:
      x-axis groups = model
      bars within each group = datasets in DATASET_ORDER
    """
    models = list(df["model"].unique())
    models.sort(key=lambda x: str(x))

    # Ensure all dataset categories exist for each model (fill missing with 0)
    grid = (
        pd.MultiIndex.from_product([models, DATASET_ORDER], names=["model", "dataset"])
        .to_frame(index=False)
        .merge(df[["model", "dataset", value_col]], on=["model", "dataset"], how="left")
        .fillna(0.0)
    )

    # positions
    n_datasets = len(DATASET_ORDER)
    n_models = len(models)
    x = range(n_models)
    total_bar_width = 0.8
    bar_width = total_bar_width / n_datasets

    # Draw bars for each dataset category
    for i, ds in enumerate(DATASET_ORDER):
        subset = grid[grid["dataset"] == ds]
        xpos = [xx + (i - (n_datasets-1)/2) * bar_width for xx in x]
        ax.bar(
            xpos,
            subset[value_col].tolist(),
            width=bar_width,
            label=DATASET_LABELS.get(ds, ds)
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(frameon=False)

    add_value_labels(ax, fmt=fmt)

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load and tag datasets
    df_cwe    = load_file(CWE_PATH);    df_cwe["dataset"]    = "cwe_analysis"
    df_sec    = load_file(SEC_PATH);    df_sec["dataset"]    = "sec_analysis"
    df_failed = load_file(FAILED_PATH); df_failed["dataset"] = "failed_analysis"

    # Analyze per row
    m_cwe    = analyze_traces(df_cwe)
    m_sec    = analyze_traces(df_sec)
    m_failed = analyze_traces(df_failed)

    # Combine
    all_metrics = pd.concat([m_cwe, m_sec, m_failed], ignore_index=True)

    # Aggregate by model × dataset
    summary = (
        all_metrics
        .groupby(["model", "dataset"], dropna=False)
        .agg({
            "repetition_ratio": "mean",
            "security_mentions": "mean",
            "trace_lines": "mean",    # NEW
            "trace_words": "mean",    # NEW
        })
        .reset_index()
    )

    # Save aggregated table
    summary.to_csv(OUT_CSV, index=False)

    # ---- Plot: one figure, four subplots (added trace length views) ----
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    grouped_bar(
        axes[0],
        summary.copy(),
        value_col="repetition_ratio",
        title="Average Repetition Ratio (Grouped by Model)",
        ylabel="Average Repetition Ratio",
        fmt=".2f"
    )

    grouped_bar(
        axes[1],
        summary.copy(),
        value_col="security_mentions",
        title="Average Security Mentions (Grouped by Model)",
        ylabel="Average Security Mentions",
        fmt=".2f"
    )

    grouped_bar(
        axes[2],
        summary.copy(),
        value_col="trace_lines",
        title="Average Trace Lines (Grouped by Model)",
        ylabel="Average # Lines",
        fmt=".0f"   # counts read better as integers
    )

    grouped_bar(
        axes[3],
        summary.copy(),
        value_col="trace_words",
        title="Average Trace Words (Grouped by Model)",
        ylabel="Average # Words",
        fmt=".0f"   # counts read better as integers
    )

    fig.suptitle("Reasoning Trace Metrics — CWE vs SEC vs FAILED (by Model)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved outputs:\n- {OUT_CSV}\n- {OUT_PNG}")

if __name__ == "__main__":
    # NOTE: This script computes aggregates and generates plots.
    # Run it once your CSVs are finalized/correct.
    main()
