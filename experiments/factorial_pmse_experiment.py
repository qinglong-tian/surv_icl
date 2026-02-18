#!/usr/bin/env python3
"""Factorial PMSE experiment for simplified_prior difficulty analysis.

Difficulty is defined as PMSE aggregated over benchmark regression models.
This script builds a factorial design over available independent prior factors,
runs benchmark regressors, and estimates factor effects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simplified_prior import SimplifiedPriorConfig, generate_simplified_prior_data


@dataclass(frozen=True)
class ExperimentConfig:
    max_conditions: int
    num_repeats: int
    num_datasets_per_condition: int
    random_seed: int
    output_dir: Path


FACTOR_LEVELS: Dict[str, List[Any]] = {
    "seq_len": [256, 512],
    "train_size": [0.5, 0.7],
    "num_features": [10, 20],
    "num_causes": [10, 20],
    "num_layers": [2, 5],
    "hidden_dim": [16, 64],
    "generation_mode": ["causal", "head", "roots"],
    "y_is_effect": [True, False],
    "in_clique": [False, True],
    "sort_features": [True, False],
    # Mapped into cfg.nonlinearities[0].
    "first_nonlinearity": ["tanh", "relu", "gelu"],
    "per_layer_activation": [False, True],
    "noise_std": [0.005, 0.03],
    "init_std": [0.5, 1.2],
    "sampling": ["normal", "uniform"],
}


def build_models() -> Dict[str, Any]:
    """Common benchmark regressors used to measure PMSE."""
    return {
        "ridge": Ridge(alpha=1.0, solver="svd"),
        "lasso": Lasso(alpha=0.001, max_iter=20000),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "gbrt": GradientBoostingRegressor(random_state=0),
        "hist_gbrt": HistGradientBoostingRegressor(random_state=0),
        "rf": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    }


def _is_valid_condition(cond: Dict[str, Any]) -> bool:
    # roots mode requires aligned dimensions.
    if cond["generation_mode"] == "roots" and int(cond["num_causes"]) != int(cond["num_features"]):
        return False
    return True


def _to_cfg_kwargs(cond: Dict[str, Any], seed: int) -> Dict[str, Any]:
    cfg = dict(cond)
    first_act = str(cfg.pop("first_nonlinearity"))
    cfg["nonlinearities"] = (first_act, "relu", "gelu")
    cfg["seed"] = int(seed)
    cfg["device"] = "cpu"
    cfg["difficulty"] = None
    return cfg


def enumerate_conditions() -> List[Dict[str, Any]]:
    names = list(FACTOR_LEVELS.keys())
    values = [FACTOR_LEVELS[name] for name in names]
    conditions: List[Dict[str, Any]] = []
    for combo in product(*values):
        cond = dict(zip(names, combo))
        if _is_valid_condition(cond):
            conditions.append(cond)
    return conditions


def sample_conditions(
    all_conditions: List[Dict[str, Any]],
    max_conditions: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    if max_conditions <= 0 or max_conditions >= len(all_conditions):
        return all_conditions
    idx = rng.choice(len(all_conditions), size=max_conditions, replace=False)
    return [all_conditions[int(i)] for i in idx]


def evaluate_one_dataset(X: np.ndarray, y: np.ndarray, train_size: int, models: Dict[str, Any]) -> List[Dict[str, Any]]:
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    rows: List[Dict[str, Any]] = []
    for model_name, model in models.items():
        try:
            m = clone(model)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            if not np.all(np.isfinite(pred)):
                continue
            pmse = float(mean_squared_error(y_test, pred))
            if not np.isfinite(pmse):
                continue
            rows.append({"model": model_name, "pmse": pmse})
        except Exception:
            continue
    return rows


def run_experiment(cfg: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    models = build_models()

    all_conditions = enumerate_conditions()
    selected = sample_conditions(all_conditions, cfg.max_conditions, rng=rng)

    print(f"Full factorial conditions (valid): {len(all_conditions)}")
    print(f"Selected conditions: {len(selected)}")
    print(f"Models: {list(models.keys())}")

    raw_rows: List[Dict[str, Any]] = []

    for cond_id, cond in enumerate(selected):
        for repeat_idx in range(cfg.num_repeats):
            seed = cfg.random_seed + cond_id * 1000 + repeat_idx
            cfg_kwargs = _to_cfg_kwargs(cond, seed=seed)
            prior_cfg = SimplifiedPriorConfig(**cfg_kwargs)
            out = generate_simplified_prior_data(prior_cfg, num_datasets=cfg.num_datasets_per_condition)
            X_batch = out["X"].numpy()
            y_batch = out["y"].numpy()
            train_sizes = out["train_sizes"].numpy()

            for ds_idx in range(X_batch.shape[0]):
                eval_rows = evaluate_one_dataset(
                    X=X_batch[ds_idx],
                    y=y_batch[ds_idx],
                    train_size=int(train_sizes[ds_idx]),
                    models=models,
                )
                for row in eval_rows:
                    base = {
                        "condition_id": cond_id,
                        "repeat": repeat_idx,
                        "dataset_idx": ds_idx,
                        "seed": int(seed),
                    }
                    base.update(cond)
                    base.update(row)
                    raw_rows.append(base)

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        raise RuntimeError("No valid benchmark rows were produced.")

    condition_model = (
        raw.groupby(["condition_id", "model"], as_index=False)
        .agg(pmse_mean=("pmse", "mean"), pmse_std=("pmse", "std"), n=("pmse", "size"))
    )

    condition_difficulty = (
        condition_model.groupby("condition_id", as_index=False)
        .agg(difficulty_pmse=("pmse_mean", "mean"), difficulty_pmse_std=("pmse_mean", "std"), n_models=("pmse_mean", "size"))
    )

    condition_factors = raw.drop_duplicates(subset=["condition_id"]) [
        ["condition_id"] + list(FACTOR_LEVELS.keys())
    ]
    condition_difficulty = condition_difficulty.merge(condition_factors, on="condition_id", how="left")

    factor_level_effects = (
        condition_difficulty.groupby(list(FACTOR_LEVELS.keys()), as_index=False)
        .agg(difficulty_pmse=("difficulty_pmse", "mean"))
    )

    # Importance from effect range across levels (main-effect screening).
    factor_importance_rows: List[Dict[str, Any]] = []
    overall_mean = float(condition_difficulty["difficulty_pmse"].mean())
    for factor in FACTOR_LEVELS:
        by_level = condition_difficulty.groupby(factor, as_index=False).agg(
            difficulty_mean=("difficulty_pmse", "mean"),
            difficulty_std=("difficulty_pmse", "std"),
            n=("difficulty_pmse", "size"),
        )
        effect_abs = float(by_level["difficulty_mean"].max() - by_level["difficulty_mean"].min())
        effect_rel = 100.0 * effect_abs / max(overall_mean, 1e-12)
        factor_importance_rows.append(
            {
                "factor": factor,
                "effect_abs": effect_abs,
                "effect_rel_pct": effect_rel,
                "overall_difficulty_mean": overall_mean,
            }
        )

    factor_importance = pd.DataFrame(factor_importance_rows).sort_values("effect_abs", ascending=False)

    # Model-specific importance.
    model_importance_rows: List[Dict[str, Any]] = []
    condition_model_wide = condition_model.merge(condition_factors, on="condition_id", how="left")
    for model_name in sorted(condition_model_wide["model"].unique()):
        subset = condition_model_wide[condition_model_wide["model"] == model_name]
        model_mean = float(subset["pmse_mean"].mean())
        for factor in FACTOR_LEVELS:
            by_level = subset.groupby(factor, as_index=False).agg(pmse_level_mean=("pmse_mean", "mean"))
            effect_abs = float(by_level["pmse_level_mean"].max() - by_level["pmse_level_mean"].min())
            effect_rel = 100.0 * effect_abs / max(model_mean, 1e-12)
            model_importance_rows.append(
                {
                    "model": model_name,
                    "factor": factor,
                    "effect_abs": effect_abs,
                    "effect_rel_pct": effect_rel,
                    "model_pmse_mean": model_mean,
                }
            )
    factor_importance_by_model = (
        pd.DataFrame(model_importance_rows).sort_values(["model", "effect_abs"], ascending=[True, False])
    )

    return {
        "raw": raw,
        "condition_model": condition_model,
        "condition_difficulty": condition_difficulty,
        "factor_level_effects": factor_level_effects,
        "factor_importance": factor_importance,
        "factor_importance_by_model": factor_importance_by_model,
        "metadata": pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "all_conditions": len(all_conditions),
                    "selected_conditions": len(selected),
                    "num_repeats": cfg.num_repeats,
                    "num_datasets_per_condition": cfg.num_datasets_per_condition,
                    "random_seed": cfg.random_seed,
                }
            ]
        ),
    }


def save_outputs(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factorial PMSE difficulty experiment.")
    parser.add_argument("--max-conditions", type=int, default=192, help="Max sampled factorial conditions (<=0 means full).")
    parser.add_argument("--num-repeats", type=int, default=2, help="Random repeats per condition.")
    parser.add_argument("--num-datasets-per-condition", type=int, default=2, help="Datasets sampled per condition/repeat.")
    parser.add_argument("--random-seed", type=int, default=20260218, help="Global random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "results" / "factorial_pmse",
        help="Directory for CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")
    args = parse_args()
    cfg = ExperimentConfig(
        max_conditions=int(args.max_conditions),
        num_repeats=int(args.num_repeats),
        num_datasets_per_condition=int(args.num_datasets_per_condition),
        random_seed=int(args.random_seed),
        output_dir=Path(args.output_dir),
    )

    tables = run_experiment(cfg)
    save_outputs(tables, cfg.output_dir)

    print("\nTop factors (overall):")
    print(tables["factor_importance"].head(10).to_string(index=False))

    print(f"\nSaved outputs to: {cfg.output_dir}")
    metadata = tables["metadata"].iloc[0].to_dict()
    print("Run metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
