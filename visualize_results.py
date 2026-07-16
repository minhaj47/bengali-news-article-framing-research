from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


RATING_LABELS = [-2, -1, 0, 1, 2]


def add_category_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["category"] = out["dataset"].astype(str).str.split("_", n=1).str[0]
    return out


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    required = {
        "dataset",
        "model",
        "coverage_vs_gold",
        "mae",
        "accuracy",
        "macro_f1",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary CSV: {sorted(missing)}")
    return add_category_column(df)


def save_sorted_tables(df: pd.DataFrame, out_dir: Path) -> None:
    detailed = df.sort_values(["dataset", "macro_f1", "mae"], ascending=[True, False, True])
    detailed.to_csv(out_dir / "all_results_sorted.csv", index=False, encoding="utf-8-sig")

    best = (
        df.sort_values(["dataset", "macro_f1", "mae"], ascending=[True, False, True])
        .groupby("dataset", as_index=False)
        .first()
    )
    best.to_csv(out_dir / "best_model_per_dataset.csv", index=False, encoding="utf-8-sig")

    model_avg = (
        df.groupby("model", as_index=False)
        .agg(
            avg_macro_f1=("macro_f1", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_mae=("mae", "mean"),
            avg_coverage=("coverage_vs_gold", "mean"),
        )
        .sort_values("avg_macro_f1", ascending=False)
    )
    model_avg.to_csv(out_dir / "model_average_metrics.csv", index=False, encoding="utf-8-sig")

    category_avg = (
        df.groupby(["category", "model"], as_index=False)
        .agg(
            avg_macro_f1=("macro_f1", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_mae=("mae", "mean"),
            avg_coverage=("coverage_vs_gold", "mean"),
        )
        .sort_values(["category", "avg_macro_f1"], ascending=[True, False])
    )
    category_avg.to_csv(out_dir / "category_model_average_metrics.csv", index=False, encoding="utf-8-sig")


def plot_model_average_bars(df: pd.DataFrame, out_dir: Path) -> None:
    model_avg = (
        df.groupby("model", as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"), mae=("mae", "mean"))
        .sort_values("macro_f1", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(model_avg["model"], model_avg["macro_f1"])
    axes[0].set_title("Average Macro F1 by Model")
    axes[0].set_ylabel("Macro F1")

    axes[1].bar(model_avg["model"], model_avg["accuracy"])
    axes[1].set_title("Average Accuracy by Model")
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(model_avg["model"], model_avg["mae"])
    axes[2].set_title("Average MAE by Model")
    axes[2].set_ylabel("MAE")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_dir / "model_average_bars.png", dpi=220)
    plt.close(fig)


def plot_metric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df.boxplot(column="macro_f1", by="model", ax=axes[0, 0])
    axes[0, 0].set_title("Macro F1 by Model")
    axes[0, 0].set_xlabel("Model")
    axes[0, 0].set_ylabel("Macro F1")

    df.boxplot(column="mae", by="model", ax=axes[0, 1])
    axes[0, 1].set_title("MAE by Model")
    axes[0, 1].set_xlabel("Model")
    axes[0, 1].set_ylabel("MAE")

    df.boxplot(column="coverage_vs_gold", by="model", ax=axes[1, 0])
    axes[1, 0].set_title("Coverage by Model")
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Coverage")

    df.boxplot(column="macro_f1", by="category", ax=axes[1, 1])
    axes[1, 1].set_title("Macro F1 by Category")
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Macro F1")

    fig.suptitle("Metric Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_distributions.png", dpi=220)
    plt.close(fig)


def _plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path, cmap: str = "viridis") -> None:
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 1.1), max(6, matrix.shape[0] * 0.35)))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_dataset_model_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    f1_pivot = df.pivot_table(index="dataset", columns="model", values="macro_f1", aggfunc="mean")
    mae_pivot = df.pivot_table(index="dataset", columns="model", values="mae", aggfunc="mean")

    f1_pivot.to_csv(out_dir / "pivot_macro_f1.csv", encoding="utf-8-sig")
    mae_pivot.to_csv(out_dir / "pivot_mae.csv", encoding="utf-8-sig")

    _plot_heatmap(f1_pivot, "Dataset x Model: Macro F1", out_dir / "heatmap_macro_f1.png", cmap="YlGnBu")
    _plot_heatmap(mae_pivot, "Dataset x Model: MAE", out_dir / "heatmap_mae.png", cmap="YlOrRd")


def plot_scatter_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, sub in df.groupby("model"):
        ax.scatter(sub["mae"], sub["macro_f1"], label=model, alpha=0.8, s=45)

    ax.set_xlabel("MAE (lower is better)")
    ax.set_ylabel("Macro F1 (higher is better)")
    ax.set_title("Trade-off: MAE vs Macro F1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_mae_vs_macro_f1.png", dpi=220)
    plt.close(fig)


def plot_model_win_counts(df: pd.DataFrame, out_dir: Path) -> None:
    best = (
        df.sort_values(["dataset", "macro_f1", "mae"], ascending=[True, False, True])
        .groupby("dataset", as_index=False)
        .first()
    )

    win_counts = best["model"].value_counts().sort_values(ascending=False)
    win_counts.to_csv(out_dir / "model_win_counts.csv", header=["wins"], encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(win_counts.index, win_counts.values)
    ax.set_title("Best-Per-Dataset Win Counts")
    ax.set_ylabel("Number of Datasets Won")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "best_model_win_counts.png", dpi=220)
    plt.close(fig)


def parse_details_name(path: Path) -> Tuple[str, str]:
    stem = path.name.replace(".csv", "")
    parts = stem.rsplit("__", 2)
    if len(parts) == 3:
        dataset, model, _ = parts
        return dataset, model
    return "unknown_dataset", "unknown_model"


def load_details(details_dir: Path) -> pd.DataFrame:
    files = sorted(details_dir.glob("*_details.csv"))
    if not files:
        return pd.DataFrame()

    pieces: List[pd.DataFrame] = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        dataset, model = parse_details_name(file)
        df["dataset"] = dataset
        df["model"] = model
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def plot_abs_error_from_details(details_df: pd.DataFrame, out_dir: Path) -> None:
    if details_df.empty or "abs_error" not in details_df.columns:
        return

    stats = (
        details_df.groupby("model", as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            n_rows=("abs_error", "count"),
        )
    )
    stats = stats.sort_values("mean_abs_error")
    stats.to_csv(out_dir / "details_abs_error_stats_by_model.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, sub in details_df.groupby("model"):
        ax.hist(sub["abs_error"].dropna(), bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], alpha=0.45, label=model)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Absolute Error |gold - pred|")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "abs_error_distribution_by_model.png", dpi=220)
    plt.close(fig)


def plot_confusion_matrices(details_df: pd.DataFrame, out_dir: Path) -> None:
    if details_df.empty:
        return
    if "gold_rating" not in details_df.columns or "llm_rating" not in details_df.columns:
        return

    details_df = details_df.dropna(subset=["gold_rating", "llm_rating"]).copy()
    if details_df.empty:
        return

    details_df["gold_rating"] = details_df["gold_rating"].astype(int)
    details_df["llm_rating"] = details_df["llm_rating"].astype(int)

    for model, sub in details_df.groupby("model"):
        cm = pd.crosstab(sub["gold_rating"], sub["llm_rating"], dropna=False)
        cm = cm.reindex(index=RATING_LABELS, columns=RATING_LABELS, fill_value=0)
        cm.to_csv(out_dir / f"confusion_matrix_{model}.csv", encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(RATING_LABELS)))
        ax.set_xticklabels(RATING_LABELS)
        ax.set_yticks(range(len(RATING_LABELS)))
        ax.set_yticklabels(RATING_LABELS)
        ax.set_xlabel("Predicted Rating")
        ax.set_ylabel("Gold Rating")
        ax.set_title(f"Confusion Matrix: {model}")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm.values[i, j]), ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix_{model}.png", dpi=240)
        plt.close(fig)


def build_all_visualizations(summary_csv: Path, details_dir: Path, out_dir: Path) -> None:
    ensure_output_dir(out_dir)

    summary_df = load_summary(summary_csv)

    save_sorted_tables(summary_df, out_dir)
    plot_model_average_bars(summary_df, out_dir)
    plot_metric_distributions(summary_df, out_dir)
    plot_dataset_model_heatmaps(summary_df, out_dir)
    plot_scatter_tradeoff(summary_df, out_dir)
    plot_model_win_counts(summary_df, out_dir)

    details_df = load_details(details_dir)
    plot_abs_error_from_details(details_df, out_dir)
    plot_confusion_matrices(details_df, out_dir)

    print("Saved visualization outputs in:", out_dir)
    print("Summary source:", summary_csv)
    print("Details source:", details_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize model-vs-gold evaluation results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evaluation_summary.csv"),
        help="Path to summary CSV (default: evaluation_summary.csv)",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("evaluation_details"),
        help="Directory containing *_details.csv files (default: evaluation_details)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation_visualizations"),
        help="Output directory for all tables and charts (default: evaluation_visualizations)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_all_visualizations(
        summary_csv=args.summary,
        details_dir=args.details_dir,
        out_dir=args.out_dir,
    )
