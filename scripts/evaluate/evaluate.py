from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from soilreg.evaluation.plotting import (
    generate_plots,
)
from sklearn.model_selection import train_test_split

# from soilreg.config.loader import load_config
import yaml
from soilreg.data.dataset import (
    SoilRegressionDataset,
    get_group_id,
)
from soilreg.data.transforms import get_valid_transforms
from soilreg.models.registry import build_model
from soilreg.preprocessing import TargetScaler





# ============================================================
# Importinf from Config
# ============================================================

def load_evaluation_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "evaluation" not in cfg:
        raise KeyError(
            "Evaluation YAML must contain a top-level 'evaluation' section."
        )

    return cfg["evaluation"]


# ============================================================
# DEVICE
# ============================================================

def resolve_device(device_name: str) -> torch.device:

    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device_name)


# ============================================================
# GROUP SPLIT
# ============================================================

def create_group_split(
    df: pd.DataFrame,
    sample_column: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    """
    Recreate the same group-based train/val/test split
    used during training.
    """

    total = train_ratio + val_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_ratio + val_ratio + test_ratio "
            "must equal 1.0"
        )

    df = df.copy()

    df["group_id"] = (
        df[sample_column]
        .astype(str)
        .apply(get_group_id)
    )

    groups = df["group_id"].unique()

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=(
            val_ratio + test_ratio
        ),
        random_state=seed,
        shuffle=True,
    )

    relative_test_ratio = (
        test_ratio
        / (val_ratio + test_ratio)
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_df = df[
        df["group_id"].isin(train_groups)
    ].copy()

    val_df = df[
        df["group_id"].isin(val_groups)
    ].copy()

    test_df = df[
        df["group_id"].isin(test_groups)
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        train_groups,
        val_groups,
        test_groups,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
):
    """
    Load either:
    - a full training checkpoint containing 'model'
    - or a raw state_dict.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
):

    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        predictions = model(images)

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_targets.append(
            targets.numpy()
        )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    return targets, predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
):

    results = {}

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    results["overall"] = {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred,
                multioutput="uniform_average",
            )
        ),
    }

    # --------------------------------------------------------
    # Per target
    # --------------------------------------------------------

    results["per_target"] = {}

    for i, target in enumerate(targets):

        mae = mean_absolute_error(
            y_true[:, i],
            y_pred[:, i],
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                y_pred[:, i],
            )
        )

        r2 = r2_score(
            y_true[:, i],
            y_pred[:, i],
        )

        results["per_target"][target] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    return results


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    sample_column: str,
    output_path: Path,
):

    predictions_df = test_df[
        [sample_column]
    ].copy()

    for i, target in enumerate(targets):

        predictions_df[
            f"{target}_actual"
        ] = y_true[:, i]

        predictions_df[
            f"{target}_predicted"
        ] = y_pred[:, i]

        predictions_df[
            f"{target}_residual"
        ] = (
            y_true[:, i]
            - y_pred[:, i]
        )

    predictions_df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_metrics(
    metrics: dict,
    targets: list[str],
):

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"MAE  : "
        f"{metrics['overall']['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{metrics['overall']['RMSE']:.4f}"
    )

    print(
        f"R²   : "
        f"{metrics['overall']['R2']:.4f}"
    )

    print("\nPer mineral:")

    for target in targets:

        values = metrics[
            "per_target"
        ][target]

        print(
            f"{target:>3} | "
            f"MAE={values['MAE']:.4f} | "
            f"RMSE={values['RMSE']:.4f} | "
            f"R²={values['R2']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main(config_path: str):

    print("=" * 70)
    print("SOIL MINERAL REGRESSION - EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    evaluation = load_evaluation_config(config_path)

    model_cfg = evaluation["model"]
    data_cfg = evaluation["data"]
    loader_cfg = evaluation["loader"]
    split_cfg = evaluation["split"]
    scaler_cfg = evaluation["scaler"]
    output_cfg = evaluation["output"]
    plot_cfg = evaluation["plots"]
    
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = resolve_device(
        evaluation["device"]
    )

    model_name = model_cfg["name"]

    print(f"\nModel  : {model_name}")
    print(f"Device : {device}")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    csv_file = Path(
        data_cfg["csv_file"]
    )

    image_dir = Path(
        data_cfg["image_dir"]
    )

    scaler_path = Path(
        scaler_cfg["path"]
    )

    output_dir = Path(
        output_cfg["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = data_cfg["targets"]

    sample_column = (
        data_cfg["sample_id_column"]
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    (
        train_df,
        val_df,
        test_df,
        train_groups,
        val_groups,
        test_groups,
    ) = create_group_split(
        df=df,
        sample_column=sample_column,
        seed=split_cfg["seed"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    print("\nDataset split:")

    print(
        f"Train images : {len(train_df)}"
    )

    print(
        f"Val images   : {len(val_df)}"
    )

    print(
        f"Test images  : {len(test_df)}"
    )

    print("\nGroups:")

    print(
        f"Train groups : {len(train_groups)}"
    )

    print(
        f"Val groups   : {len(val_groups)}"
    )

    print(
        f"Test groups  : {len(test_groups)}"
    )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler = TargetScaler.load(
        scaler_path
    )

    # --------------------------------------------------------
    # Scale test targets
    # --------------------------------------------------------

    scaled_test_df = scaler.transform(
        test_df.copy(),
        targets,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = SoilRegressionDataset(
        dataframe=scaled_test_df,
        image_dir=image_dir,
        targets=targets,
        sample_column=sample_column,
        image_extension=(
            data_cfg["image_extension"]
        ),
        transforms=get_valid_transforms(
            loader_cfg["image_size"]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=(
            loader_cfg["pin_memory"]
            and device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        model_name=model_name,
        num_outputs=model_cfg["num_outputs"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )

    checkpoint_path = (
        model_cfg["checkpoint"]["path"]
    )

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning inference...")

    y_true_scaled, y_pred_scaled = predict(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Inverse scaling
    # --------------------------------------------------------

    y_true = scaler.inverse_transform(
        y_true_scaled
    )

    y_pred = scaler.inverse_transform(
        y_pred_scaled
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred,
        targets,
    )

    print_metrics(
        metrics,
        targets,
    )
    
    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    if plot_cfg["enabled"]:

        plots_dir = (
            output_dir
            / plot_cfg["directory"]
        )

        print("\nGenerating plots...")

        generate_plots(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            targets=targets,
            output_dir=plots_dir,
            actual_vs_predicted=(
                plot_cfg["actual_vs_predicted"]
            ),
            residuals=(
                plot_cfg["residuals"]
            ),
            metrics_enabled=(
                plot_cfg["metrics"]
            ),
        )

        print(
            f"Plots saved to: {plots_dir}"
        )
    

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = (
        output_dir
        / output_cfg["predictions_file"]
    )

    save_predictions(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        targets=targets,
        sample_column=sample_column,
        output_path=predictions_path,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        output_dir
        / output_cfg["metrics_file"]
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)

    print("\nEvaluation complete.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate soil mineral regression model"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation.yaml",
        help="Path to evaluation YAML configuration",
    )

    args = parser.parse_args()

    main(args.config)