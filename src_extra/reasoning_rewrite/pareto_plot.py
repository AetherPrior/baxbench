import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl


PARENT_DIR = "./new_interventions"
MODEL = "openai-gpt-oss-120b_high"

model_name_or_path = "openai/gpt-oss-120b"
data_paths = {
    "prelim": f"{PARENT_DIR}/prelim/{MODEL}_all_analysis.csv",
    "final": f"{PARENT_DIR}/final_enumerate/{MODEL}_all_analysis.csv",
    "prelim+scaffold": f"{PARENT_DIR}/prelim_scaffold/{MODEL}_all_analysis.csv",
    "baseline_none": f"{PARENT_DIR}/none/none/{MODEL}_all_analysis.csv",
    "baseline_oracle": f"{PARENT_DIR}/none/specific/{MODEL}_all_analysis.csv",
    "baseline_generic": f"{PARENT_DIR}/none/generic/{MODEL}_all_analysis.csv",
}

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().replace("<think>", "").replace("</think>", "").strip()

def get_token_count(path, tokenizer) -> float:
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(subset=['gen_text'])
    token_counts = []
    for text in df['gen_text']:
        text = _clean_text(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
    token_counts = np.array(token_counts)
    return float(np.mean(token_counts)) if len(token_counts) else 0.0

# NEW: return full distribution of token counts for a file
def get_token_counts(path, tokenizer) -> list[int]:
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(subset=['gen_text'])
    counts = []
    for text in df['gen_text']:
        text = _clean_text(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        counts.append(len(tokens))
    return counts

def plot_2d(token_counts: list[int], pass_scores: list[float], sec_pass_scores: list[float], type_strs: list[str], annotate: bool = True):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(token_counts, sec_pass_scores, c=pass_scores, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Pass@1 Score')
    for i, type_str in enumerate(type_strs):
        ax.annotate(type_str, (token_counts[i], sec_pass_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

    if annotate:
        # annotate pass_score, sec_pass_score, mean token count
        for i in range(len(type_strs)):
            ax.annotate(
                f"P@1: {pass_scores[i]:.2f}\nSP@1: {sec_pass_scores[i]:.2f}\nTC: {token_counts[i]:.0f}",
                xy=(token_counts[i], sec_pass_scores[i]),
                xytext=(0, 25),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.8,
            )
        
    ax.set_xlabel('Average Token Count of Reasoning')
    ax.set_ylabel('Sec_Pass@1 Score')
    ax.set_title(f'Token Count vs Sec_Pass@1')
    ax.grid(True, linestyle='--', alpha=0.3)
    Path(PARENT_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{PARENT_DIR}/pareto_plot.png", bbox_inches='tight', dpi=200)
    plt.close()

# NEW: horizontal violin plot of token-count distributions per type
def plot_token_violins(token_counts_by_type: dict[str, list[int]], *, savepath: str):
    """
    Draws a horizontal violin for each type showing the distribution of token counts.
    Also overlays a marker for the mean token count per type.
    """
    # Build a long-form DataFrame: columns = ['type', 'token_count']
    rows = []
    for t, counts in token_counts_by_type.items():
        for c in counts:
            rows.append((t, c))
    plot_df = pd.DataFrame(rows, columns=['type', 'token_count'])

    # Sort types by mean token count (descending) for a nice layout
    type_order = (
        plot_df.groupby('type')['token_count']
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=plot_df,
        y='type',
        x='token_count',
        orient='h',
        order=type_order,
        cut=0,
        inner=None,       # cleaner silhouette; we'll add means separately
        linewidth=1
    )

    # Overlay mean markers
    means = plot_df.groupby('type')['token_count'].mean()
    for i, t in enumerate(type_order):
        plt.scatter(means[t], i, s=60, edgecolor='black', zorder=3)


    plt.xlabel("Token Count (distribution)")
    plt.ylabel("Type")
    plt.title("Reasoning Token Count Distributions by Type (Horizontal Violins)")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    Path(PARENT_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight', dpi=200)
    plt.close()

# --- NEW: 2D plot with horizontal violins placed at each sec_pass y ---
def plot_2d_with_horizontal_violins(
    token_counts_by_type: dict[str, list[int]],
    pass_scores: dict[str, float],
    sec_pass_scores: dict[str, float],
    *,
    savepath: str,
    point_alpha: float = 0.9,
    violin_alpha: float = 0.35,
    violin_width: float = 0.035,   # thickness along the y-axis (in y-data units)
    annotate: bool = True,
):
    """
    For each type, draws a HORIZONTAL violin at y = sec_pass[type], over x = token_count distribution.
    The point's hue is the pass score; violins are filled with the same colormap color.
    """
    # Prepare colormap for pass score hue
    cmap = mpl.cm.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=min(pass_scores.values()), vmax=max(pass_scores.values()))

    # Collect global x-lims for nicer layout
    all_counts = [c for counts in token_counts_by_type.values() for c in counts]
    if not all_counts:
        raise ValueError("No token counts found to plot.")
    x_min, x_max = float(np.percentile(all_counts, 1)), float(np.percentile(all_counts, 99))
    # small padding
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    x_min, x_max = max(0.0, x_min - pad), x_max + pad

    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw a violin and a mean-dot for each type at y = sec_pass
    for t, counts in token_counts_by_type.items():
        if len(counts) == 0:
            continue
        y = sec_pass_scores[t]
        color = cmap(norm(pass_scores[t]))
        # Matplotlib's violinplot expects a sequence of datasets; we pass [counts]
        v = ax.violinplot(
            [counts],
            positions=[y],
            vert=False,
            widths=violin_width,   # thickness in y-units
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        # Color the violin
        for body in v['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(violin_alpha)

        # Overlay the mean as a point (same hue)
        mean_x = float(np.mean(counts))
        ax.scatter(mean_x, y, s=60, c=[color], edgecolor="black", linewidths=0.6, alpha=point_alpha)

        # Optional label slightly above the mean point
        if annotate:
            # annotate pass_score, sec_pass_score, mean token count
            ax.annotate(
                f"{t}\nP@1: {pass_scores[t]:.2f}\nSP@1: {y:.2f}\nTC: {mean_x:.0f}",
                xy=(mean_x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.8,
            )

    # Axes, grid, and colorbar
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Reasoning Token Count (distribution via horizontal violins)")
    ax.set_ylabel("Sec_Pass@1 Score")
    ax.set_title("Token Count Distributions vs Sec_Pass@1 (Hue = Pass@1)")

    # neat x-grid for reading token counts
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Continuous colorbar for pass score
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Pass@1 Score")

    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    all_token_counts = []
    all_pass_scores = []
    all_sec_pass_scores = []
    all_type_strs = []

    pass_scores = {'prelim': 0.63, 'final': 0.72, 'prelim+scaffold': 0.63, 'baseline_none': 0.71, 'baseline_oracle': 0.60, 'baseline_generic': 0.62}
    sec_pass_scores = {'prelim': 0.47, 'final': 0.51, 'prelim+scaffold': 0.51, 'baseline_none': 0.43, 'baseline_oracle': 0.54, 'baseline_generic': 0.42}

    # For the scatter (means)
    for type_str, path in data_paths.items():
        token_count = get_token_count(path, tokenizer)
        all_token_counts.append(token_count)
        all_pass_scores.append(pass_scores[type_str])
        all_sec_pass_scores.append(sec_pass_scores[type_str])
        all_type_strs.append(type_str)

    plot_2d(all_token_counts, all_pass_scores, all_sec_pass_scores, all_type_strs)

    token_counts_by_type = {}
    for type_str, path in data_paths.items():
        token_counts_by_type[type_str] = get_token_counts(path, tokenizer)

    plot_2d_with_horizontal_violins(
        token_counts_by_type=token_counts_by_type,
        pass_scores=pass_scores,
        sec_pass_scores=sec_pass_scores,
        savepath=f"{PARENT_DIR}/pareto_violin2d.png",
        violin_width=0.035,   # tweak if violins look too thick/thin (y is 0..1)
    )
