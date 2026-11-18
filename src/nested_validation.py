
"""
Nested depth-blocked cross-validation for well-log ML research.

Outer CV: 5 contiguous depth blocks.
Inner CV: 3 contiguous depth blocks within outer training data.

All data-dependent preprocessing is fitted inside CV folds.

This module does not load or export confidential well data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.depth_blocked_validation import DepthBlockedKFold


@dataclass
class NestedCVResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_summary: pd.DataFrame


def build_models(random_state=42):
    """Return model pipelines and compact parameter grids."""

    models = {
        "RandomForest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    random_state=random_state,
                    n_jobs=1,
                )),
            ]),
            {
                "model__n_estimators": [100],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "ExtraTrees": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(
                    random_state=random_state,
                    n_jobs=1,
                )),
            ]),
            {
                "model__n_estimators": [100],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "Ridge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            {
                "model__alpha": [0.1, 1.0, 10.0],
            },
        ),
        "HistGradientBoosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(
                    random_state=random_state,
                    early_stopping=False,
                )),
            ]),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [100],
                "model__max_leaf_nodes": [7, 15],
                "model__min_samples_leaf": [10],
            },
        ),
    }

    return models


def run_nested_depth_cv(
    data,
    feature_columns,
    target_column,
    depth_column,
    outer_splits=5,
    inner_splits=3,
    random_state=42,
):
    """
    Run nested depth-blocked CV on a single-well dataset.

    Target should be log10(permeability_mD), prepared upstream.

    The input is sorted by depth before creating folds.
    Depth itself is not automatically included as a predictor.

    Returns fold-level metrics, pooled OOF metrics,
    and row-aligned OOF predictions for each model.
    """

    required = set(feature_columns) | {
        target_column,
        depth_column,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not feature_columns:
        raise ValueError("At least one feature is required.")

    if target_column in feature_columns:
        raise ValueError("Target cannot be a predictor.")

    if depth_column in feature_columns:
        raise ValueError(
            "Depth cannot be a predictor in this workflow."
        )

    df = data.copy()

    # Preserve the original row position for traceability.
    df["_original_position"] = np.arange(len(df))

    numeric_columns = list(
        dict.fromkeys(
            feature_columns + [target_column, depth_column]
        )
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if not np.isfinite(df[depth_column].to_numpy()).all():
        raise ValueError("Depth contains non-finite values.")

    if not np.isfinite(df[target_column].to_numpy()).all():
        raise ValueError("Target contains non-finite values.")

    feature_values = df[feature_columns].to_numpy(dtype=float)

    if np.isinf(feature_values).any():
        raise ValueError("Features contain infinity.")

    if np.isnan(feature_values).all(axis=0).any():
        raise ValueError(
            "At least one feature contains only missing values."
        )

    df = df.sort_values(
        depth_column,
        kind="stable",
    ).reset_index(drop=True)

    if df[depth_column].duplicated().any():
        raise ValueError(
            "Duplicate depths found. Resolve depth alignment "
            "before running validation."
        )

    X = df[feature_columns]
    y = df[target_column].to_numpy(dtype=float)

    if len(df) < outer_splits + inner_splits:
        raise ValueError("Insufficient observations for nested CV.")

    outer_cv = DepthBlockedKFold(n_splits=outer_splits)

    model_specs = build_models(random_state=random_state)

    prediction_records = []
    metric_records = []

    for model_name, (pipeline, param_grid) in model_specs.items():

        oof_predictions = np.full(len(df), np.nan)

        for fold_number, (train_idx, test_idx) in enumerate(
            outer_cv.split(X),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y[test_idx]

            # train_idx remains in ascending depth order.
            inner_cv = DepthBlockedKFold(
                n_splits=inner_splits
            )

            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            oof_predictions[test_idx] = y_pred

            fold_mae = mean_absolute_error(
                y_test,
                y_pred,
            )

            fold_rmse = np.sqrt(
                mean_squared_error(y_test, y_pred)
            )

            metric_records.append({
                "model": model_name,
                "fold": fold_number,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "mae_log10": fold_mae,
                "rmse_log10": fold_rmse,
                "best_params": str(search.best_params_),
            })

            print(
                f"{model_name} | Fold {fold_number} | "
                f"MAE={fold_mae:.4f}"
            )

        if np.isnan(oof_predictions).any():
            raise RuntimeError("Some OOF predictions are missing.")

        for position, prediction in enumerate(oof_predictions):
            prediction_records.append({
                "model": model_name,
                "sorted_position": position,
                "original_position": int(
                    df.loc[position, "_original_position"]
                ),
                "actual_log10": y[position],
                "predicted_log10": float(prediction),
            })

    predictions = pd.DataFrame(prediction_records)
    fold_metrics = pd.DataFrame(metric_records)

    summary_records = []

    for model_name, group in predictions.groupby("model"):
        actual = group["actual_log10"].to_numpy()
        predicted = group["predicted_log10"].to_numpy()

        summary_records.append({
            "model": model_name,
            "n": len(group),
            "pooled_oof_mae_log10": mean_absolute_error(
                actual, predicted
            ),
            "pooled_oof_rmse_log10": np.sqrt(
                mean_squared_error(actual, predicted)
            ),
        })

    model_summary = pd.DataFrame(
        summary_records
    ).sort_values("pooled_oof_mae_log10")

    return NestedCVResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        model_summary=model_summary,
    )
