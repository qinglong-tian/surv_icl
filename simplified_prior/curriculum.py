"""Curriculum samplers for the simplified prior.

This module explicitly separates stage-dependent and stage-invariant sampling:
- Stage-dependent: generation_mode, num_layers, hidden_dim
- Stage-invariant: all other SimplifiedPriorConfig fields
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np

from .generator import SimplifiedPriorConfig, generate_simplified_prior_data


@dataclass(frozen=True)
class CurriculumBounds:
    num_layers_min: int
    num_layers_max: int
    hidden_dim_min: int
    hidden_dim_max: int


STAGE_DEPENDENT_FACTORS = {"generation_mode", "num_layers", "hidden_dim"}

# These are derived from generation_mode in SimplifiedPriorConfig.
_DERIVED_MODE_FACTORS = {"is_causal", "noncausal_feature_source"}

STAGE_CONTROLLED_FACTORS = STAGE_DEPENDENT_FACTORS | _DERIVED_MODE_FACTORS
STAGE_INVARIANT_FACTORS = frozenset(f.name for f in fields(SimplifiedPriorConfig)) - STAGE_CONTROLLED_FACTORS


def _is_sequence_like(x: object) -> bool:
    if isinstance(x, (str, bytes, dict)):
        return False
    return isinstance(x, (list, tuple, np.ndarray))


def _sample_value(spec: object, rng: np.random.Generator) -> object:
    """Sample one value from a stationary sampler spec.

    Supported specs:
    - Callable: f(rng) -> value
    - Sequence: uniform over values
    - Scalar/object: constant
    """
    if callable(spec):
        return spec(rng)
    if _is_sequence_like(spec):
        values = list(spec)  # type: ignore[arg-type]
        if len(values) == 0:
            raise ValueError("Stationary sampler list cannot be empty.")
        return values[int(rng.integers(0, len(values)))]
    return spec


def sample_stage_invariant_hyperparameters(
    base_cfg: SimplifiedPriorConfig,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample only stage-invariant config fields.

    `stationary_sampler` can override any stage-invariant key. Stage-dependent keys
    (`generation_mode`, `num_layers`, `hidden_dim`) and derived mode keys
    (`is_causal`, `noncausal_feature_source`) are disallowed.
    """
    if rng is None:
        rng = np.random.default_rng()

    cfg_dict = asdict(base_cfg)
    sampler = stationary_sampler or {}

    for key, spec in sampler.items():
        if key in STAGE_CONTROLLED_FACTORS:
            raise ValueError(f"'{key}' is stage-dependent/derived and cannot be in stationary_sampler.")
        if key not in cfg_dict:
            raise ValueError(f"Unknown config key in stationary_sampler: '{key}'")
        cfg_dict[key] = _sample_value(spec, rng)

    return cfg_dict


def is_causal_false_probability(stage_idx: int, total_stages: int) -> float:
    """P(non-causal) schedule at stage s in {1, ..., K}: 1 - (s-1)/(2K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    prob = 1.0 - ((stage_idx - 1) / (2.0 * total_stages))
    return float(np.clip(prob, 0.0, 1.0))


def generation_mode_probabilities(
    stage_idx: int,
    total_stages: int,
    p_roots_given_noncausal: float = 0.5,
) -> Dict[str, float]:
    """Return stage-wise probabilities for generation_mode.

    p_noncausal is stage-dependent via `is_causal_false_probability`.
    p_roots_given_noncausal is a global curriculum hyperparameter controlling the
    head/roots split inside non-causal mass.
    """
    if not (0.0 <= float(p_roots_given_noncausal) <= 1.0):
        raise ValueError("p_roots_given_noncausal must be in [0, 1].")

    p_noncausal = is_causal_false_probability(stage_idx=stage_idx, total_stages=total_stages)
    p_causal = 1.0 - p_noncausal
    p_roots = p_noncausal * float(p_roots_given_noncausal)
    p_head = p_noncausal * (1.0 - float(p_roots_given_noncausal))
    return {"causal": float(p_causal), "head": float(p_head), "roots": float(p_roots)}


def _sample_generation_mode(
    stage_idx: int,
    total_stages: int,
    num_causes: int,
    num_features: int,
    p_roots_given_noncausal: float,
    rng: np.random.Generator,
) -> str:
    probs = generation_mode_probabilities(
        stage_idx=stage_idx,
        total_stages=total_stages,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
    )

    if int(num_causes) == int(num_features):
        modes = ["causal", "head", "roots"]
    else:
        modes = ["causal", "head"]

    unnormalized = np.array([probs[m] for m in modes], dtype=np.float64)
    total = float(unnormalized.sum())
    if total <= 0.0:
        weights = np.full_like(unnormalized, 1.0 / len(unnormalized))
    else:
        weights = unnormalized / total
    return str(rng.choice(modes, p=weights))


def stage_upper_limit(stage_idx: int, total_stages: int, lo: int, hi: int) -> int:
    """Linear growth of upper bound from lo to hi over stages."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    if lo > hi:
        raise ValueError("Lower bound must be <= upper bound.")

    if total_stages == 1:
        return int(hi)
    frac = (stage_idx - 1) / (total_stages - 1)
    return int(round(lo + frac * (hi - lo)))


def stage_linear_probability(stage_idx: int, total_stages: int, start: float, end: float) -> float:
    """Linear schedule from `start` (stage 1) to `end` (stage K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("start and end must be in [0, 1].")

    if total_stages == 1:
        return float(end)
    frac = (stage_idx - 1) / (total_stages - 1)
    return float((1.0 - frac) * start + frac * end)


def sample_stage_dependent_hyperparameters(
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_causes: int,
    num_features: int,
    p_roots_given_noncausal: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Sample only stage-dependent config fields."""
    layer_upper = stage_upper_limit(
        stage_idx=stage_idx,
        total_stages=total_stages,
        lo=int(bounds.num_layers_min),
        hi=int(bounds.num_layers_max),
    )
    hidden_upper = stage_upper_limit(
        stage_idx=stage_idx,
        total_stages=total_stages,
        lo=int(bounds.hidden_dim_min),
        hi=int(bounds.hidden_dim_max),
    )

    num_layers = int(rng.integers(int(bounds.num_layers_min), int(layer_upper) + 1))
    hidden_dim = int(rng.integers(int(bounds.hidden_dim_min), int(hidden_upper) + 1))
    generation_mode = _sample_generation_mode(
        stage_idx=stage_idx,
        total_stages=total_stages,
        num_causes=int(num_causes),
        num_features=int(num_features),
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        rng=rng,
    )

    return {
        "generation_mode": generation_mode,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
    }


def sample_curriculum_config(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimplifiedPriorConfig:
    """Sample one stage config with explicit dependent/invariant split."""
    if rng is None:
        rng = np.random.default_rng()

    cfg_dict = sample_stage_invariant_hyperparameters(
        base_cfg=base_cfg,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    stage_cfg = sample_stage_dependent_hyperparameters(
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        num_causes=int(cfg_dict["num_causes"]),
        num_features=int(cfg_dict["num_features"]),
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        rng=rng,
    )

    cfg_dict["difficulty"] = None
    cfg_dict.update(stage_cfg)
    return SimplifiedPriorConfig(**cfg_dict)


def generate_curriculum_stage_batch(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_datasets: int,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]]:
    """Sample a stage config and generate datasets with that config."""
    stage_cfg = sample_curriculum_config(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    batch = generate_simplified_prior_data(stage_cfg, num_datasets=num_datasets)
    return stage_cfg, batch
