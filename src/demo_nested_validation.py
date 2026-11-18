
"""Reproducible synthetic demonstration of nested depth-blocked CV."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.nested_validation import run_nested_depth_cv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "synthetic_demo"


def create_synthetic_data(n=150, seed=42):
    """Generate artificial well-log-like observations."""
    rng = np.random.default_rng(seed)

    depth = np.arange(n, dtype=float)
    gr = 70 + 20 * np.sin(depth / 18) + rng.normal(0, 5, n)
    dt = 85 + 0.25 * gr + rng.normal(0, 4, n)
    rt = np.exp(2.5 - 0.012 * gr + rng.normal(0, 0.2, n))

    log_perm = (
        2.0
        - 0.018 * gr
        + 0.012 * dt
        + 0.15 * np.log10(rt)
        + rng.normal(0, 0.15, n)
    )

    return pd.DataFrame({
        "depth": depth,
        "GR": gr,
        "DT": dt,
        "RT": rt,
        "log10_permeability": log_perm,
    })


def plot_actual_vs_predicted(predictions, output_path):
    """Plot pooled out-of-fold predictions for each model."""
    models = predictions["model"].unique()

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9), constrained_layout=True
    )

    for ax, model in zip(axes.flat, models):
        group = predictions[predictions["model"] == model]

        actual = group["actual_log10"]
        predicted = group["predicted_log10"]

        ax.scatter(actual, predicted, s=16, alpha=0.65)

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_title(model)
        ax.set_xlabel("Synthetic actual log10(k / mD)")
        ax.set_ylabel("OOF predicted log10(k / mD)")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Synthetic Well-Log ML | Actual vs Predicted",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residuals(predictions, output_path):
    """Plot prediction residuals against synthetic actual values."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model, group in predictions.groupby("model"):
        residual = (
            group["predicted_log10"] - group["actual_log10"]
        )

        ax.scatter(
            group["actual_log10"],
            residual,
            s=14,
            alpha=0.45,
            label=model,
        )

    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Synthetic actual log10(k / mD)")
    ax.set_ylabel("Residual: predicted - actual")
    ax.set_title("Synthetic Well-Log ML | OOF Residuals")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_comparison(summary, output_path):
    """Compare pooled out-of-fold MAE across models."""
    ordered = summary.sort_values("pooled_oof_mae_log10")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        ordered["model"],
        ordered["pooled_oof_mae_log10"],
    )
    ax.invert_yaxis()

    ax.set_xlabel("Pooled OOF MAE [log10(k / mD)]")
    ax.set_title("Synthetic Well-Log ML | Model Comparison")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = create_synthetic_data()

    result = run_nested_depth_cv(
        data=df,
        feature_columns=["GR", "DT", "RT"],
        target_column="log10_permeability",
        depth_column="depth",
        outer_splits=5,
        inner_splits=3,
    )

    # Outputs contain synthetic observations only.
    result.model_summary.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    result.fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    result.predictions.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        result.predictions,
        OUTPUT_DIR / "actual_vs_predicted.png",
    )

    plot_residuals(
        result.predictions,
        OUTPUT_DIR / "residuals.png",
    )

    plot_model_comparison(
        result.model_summary,
        OUTPUT_DIR / "model_comparison.png",
    )

    print("\nSYNTHETIC DATA | MODEL COMPARISON")
    print(result.model_summary.to_string(index=False))

    print(f"\nSaved outputs: {OUTPUT_DIR}")
    print("No confidential research data were used.")


if __name__ == "__main__":
    main()
