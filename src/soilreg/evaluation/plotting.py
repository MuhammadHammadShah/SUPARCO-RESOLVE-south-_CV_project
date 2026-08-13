from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    output_dir: Path,
) -> None:
    """
    Create one Actual vs Predicted plot for each target.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i, target in enumerate(targets):

        actual = y_true[:, i]
        predicted = y_pred[:, i]

        min_value = min(
            actual.min(),
            predicted.min(),
        )

        max_value = max(
            actual.max(),
            predicted.max(),
        )

        plt.figure(figsize=(7, 7))

        plt.scatter(
            actual,
            predicted,
            alpha=0.7,
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
            linewidth=2,
            label="Ideal prediction",
        )

        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        plt.title(
            f"{target} — Actual vs Predicted"
        )

        plt.legend()
        plt.grid(
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / f"{target}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    output_dir: Path,
) -> None:
    """
    Create one residual plot for each target.

    Residual = Actual - Predicted
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i, target in enumerate(targets):

        actual = y_true[:, i]
        predicted = y_pred[:, i]

        residuals = (
            actual - predicted
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            predicted,
            residuals,
            alpha=0.7,
        )

        plt.axhline(
            0,
            linestyle="--",
            linewidth=2,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Residual")

        plt.title(
            f"{target} — Residual Plot"
        )

        plt.grid(
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / f"{target}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def plot_metric(
    metrics: dict,
    targets: list[str],
    metric_name: str,
    output_path: Path,
) -> None:
    """
    Create a bar chart for one metric across all targets.
    """

    values = [
        metrics["per_target"][target][metric_name]
        for target in targets
    ]

    plt.figure(
        figsize=(9, 6),
    )

    bars = plt.bar(
        targets,
        values,
    )

    plt.xlabel("Mineral")
    plt.ylabel(metric_name)

    plt.title(
        f"{metric_name} by Mineral"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_metrics(
    metrics: dict,
    targets: list[str],
    output_dir: Path,
) -> None:
    """
    Create MAE, RMSE and R² plots.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric_name in (
        "MAE",
        "RMSE",
        "R2",
    ):

        plot_metric(
            metrics=metrics,
            targets=targets,
            metric_name=metric_name,
            output_path=(
                output_dir
                / f"{metric_name.lower()}.png"
            ),
        )


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    targets: list[str],
    output_dir: Path,
    actual_vs_predicted: bool = True,
    residuals: bool = True,
    metrics_enabled: bool = True,
) -> None:
    """
    Generate all configured evaluation plots.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if actual_vs_predicted:

        plot_actual_vs_predicted(
            y_true=y_true,
            y_pred=y_pred,
            targets=targets,
            output_dir=(
                output_dir
                / "actual_vs_predicted"
            ),
        )

    if residuals:

        plot_residuals(
            y_true=y_true,
            y_pred=y_pred,
            targets=targets,
            output_dir=(
                output_dir
                / "residuals"
            ),
        )

    if metrics_enabled:

        plot_metrics(
            metrics=metrics,
            targets=targets,
            output_dir=(
                output_dir
                / "metrics"
            ),
        )