"""Curriculum samplers for the simplified prior.

This module separates stage-invariant and stage-dependent sampling while
supporting smooth probability annealing and extensible future factors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .generator import SimplifiedPriorConfig, generate_simplified_prior_data


@dataclass(frozen=True)
class CurriculumBounds:
    num_layers_min: int
    num_layers_max: int
    hidden_dim_min: int
    hidden_dim_max: int


@dataclass(frozen=True)
class AnnealedCategoricalSchedule:
    """Smooth categorical schedule from start probs to end probs."""

    start_probs: Mapping[str, float]
    end_probs: Mapping[str, float]
    gamma: float = 1.5

    def __post_init__(self) -> None:
        if float(self.gamma) <= 0.0:
            raise ValueError("gamma must be > 0.")
        for name, probs in (("start_probs", self.start_probs), ("end_probs", self.end_probs)):
            if len(probs) == 0:
                raise ValueError(f"{name} must be non-empty.")
            for k, v in probs.items():
                if float(v) < 0.0:
                    raise ValueError(f"{name} has negative probability for '{k}': {v}")
            total = float(sum(float(v) for v in probs.values()))
            if total <= 0.0:
                raise ValueError(f"{name} must sum to > 0.")

    def probabilities(self, progress: float) -> Dict[str, float]:
        p = float(np.clip(progress, 0.0, 1.0))
        t = float(p**float(self.gamma))
        keys = sorted(set(self.start_probs) | set(self.end_probs))

        probs: Dict[str, float] = {}
        for k in keys:
            s = float(self.start_probs.get(k, 0.0))
            e = float(self.end_probs.get(k, 0.0))
            probs[k] = (1.0 - t) * s + t * e

        total = float(sum(probs.values()))
        if total <= 0.0:
            n = len(probs)
            return {k: 1.0 / n for k in probs}
        return {k: float(v / total) for k, v in probs.items()}

    def sample(
        self,
        progress: float,
        rng: np.random.Generator,
        allowed_levels: Optional[Sequence[str]] = None,
    ) -> str:
        probs = self.probabilities(progress)

        if allowed_levels is None:
            levels = sorted(probs)
            weights = np.array([probs[k] for k in levels], dtype=np.float64)
        else:
            levels = [str(x) for x in allowed_levels]
            if len(levels) == 0:
                raise ValueError("allowed_levels must be non-empty when provided.")
            weights = np.array([float(probs.get(k, 0.0)) for k in levels], dtype=np.float64)
            total = float(weights.sum())
            if total <= 0.0:
                weights = np.full_like(weights, 1.0 / len(weights))
            else:
                weights = weights / total

        return str(rng.choice(levels, p=weights))


@dataclass(frozen=True)
class GenerationModeSchedule:
    """Default smooth schedule for generation_mode."""

    start_probs: Mapping[str, float] = field(
        default_factory=lambda: {"head": 0.75, "causal": 0.20, "roots": 0.05}
    )
    end_probs: Mapping[str, float] = field(
        default_factory=lambda: {"head": 0.15, "causal": 0.30, "roots": 0.55}
    )
    gamma: float = 1.5

    def as_annealed_schedule(self) -> AnnealedCategoricalSchedule:
        return AnnealedCategoricalSchedule(
            start_probs=self.start_probs,
            end_probs=self.end_probs,
            gamma=float(self.gamma),
        )


@dataclass(frozen=True)
class SmoothIntegerSchedule:
    """Continuous schedule for integer-valued factors.

    For progress p in [0, 1], define:
    - p_tilde = p^gamma
    - x(p) = lo + (hi - lo) * p_tilde

    Sampling uses stochastic rounding around x(p), so expected value changes
    continuously as stage advances, and per-stage changes shrink as K grows.
    """

    lo: int
    hi: int
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if int(self.lo) > int(self.hi):
            raise ValueError("lo must be <= hi.")
        if float(self.gamma) <= 0.0:
            raise ValueError("gamma must be > 0.")

    def expected_value(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        t = float(p**float(self.gamma))
        return float(int(self.lo) + (int(self.hi) - int(self.lo)) * t)

    def sample(self, progress: float, rng: np.random.Generator) -> int:
        x = self.expected_value(progress=progress)
        lo_i = int(np.floor(x))
        hi_i = int(np.ceil(x))
        lo_i = int(np.clip(lo_i, int(self.lo), int(self.hi)))
        hi_i = int(np.clip(hi_i, int(self.lo), int(self.hi)))
        if hi_i <= lo_i:
            return lo_i
        p_hi = float(np.clip(x - lo_i, 0.0, 1.0))
        draw = float(rng.random())
        return hi_i if draw < p_hi else lo_i


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
    """Sample one value from a stationary sampler spec."""
    if callable(spec):
        return spec(rng)
    if _is_sequence_like(spec):
        values = list(spec)  # type: ignore[arg-type]
        if len(values) == 0:
            raise ValueError("Sampler list cannot be empty.")
        return values[int(rng.integers(0, len(values)))]
    return spec


def _resolve_progress(
    stage_idx: int,
    total_stages: int,
    progress: Optional[float] = None,
) -> float:
    if progress is not None:
        return float(np.clip(progress, 0.0, 1.0))
    return stage_progress(stage_idx=stage_idx, total_stages=total_stages)


def _linear_int_upper_from_progress(progress: float, lo: int, hi: int) -> int:
    if lo > hi:
        raise ValueError("Lower bound must be <= upper bound.")
    p = float(np.clip(progress, 0.0, 1.0))
    return int(round(lo + p * (hi - lo)))


def _sample_from_stage_sampler(
    spec: object,
    progress: float,
    current_values: Dict[str, object],
    rng: np.random.Generator,
) -> object:
    """Sample value from an extra stage sampler spec.

    Supported specs:
    - AnnealedCategoricalSchedule: sampled category
    - Callable with one of signatures:
      f(progress, current_values, rng), f(progress, rng), or f(rng)
    - Sequence: uniform over values
    - Scalar/object: constant
    """
    if isinstance(spec, AnnealedCategoricalSchedule):
        return spec.sample(progress=progress, rng=rng)

    if callable(spec):
        try:
            return spec(progress, current_values, rng)
        except TypeError:
            try:
                return spec(progress, rng)
            except TypeError:
                return spec(rng)

    if _is_sequence_like(spec):
        values = list(spec)  # type: ignore[arg-type]
        if len(values) == 0:
            raise ValueError("Extra stage sampler list cannot be empty.")
        return values[int(rng.integers(0, len(values)))]

    return spec


def stage_progress(stage_idx: int, total_stages: int) -> float:
    """Map stage index to progress in [0, 1]."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    if total_stages == 1:
        return 1.0
    return float((stage_idx - 1) / (total_stages - 1))


def sample_stage_invariant_hyperparameters(
    base_cfg: SimplifiedPriorConfig,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample only stage-invariant config fields."""
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
    """Legacy schedule: P(non-causal) at stage s in {1, ..., K}: 1 - (s-1)/(2K)."""
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
    mode_schedule: Optional[GenerationModeSchedule] = None,
    progress: Optional[float] = None,
) -> Dict[str, float]:
    """Return stage-wise probabilities for generation_mode.

    If `mode_schedule` is provided, smooth annealing is used.
    Otherwise, a legacy probability rule is used for backward compatibility.
    """
    p = _resolve_progress(stage_idx=stage_idx, total_stages=total_stages, progress=progress)

    if mode_schedule is not None:
        return mode_schedule.as_annealed_schedule().probabilities(progress=p)

    if not (0.0 <= float(p_roots_given_noncausal) <= 1.0):
        raise ValueError("p_roots_given_noncausal must be in [0, 1].")

    # Backward-compatible behavior.
    p_noncausal = is_causal_false_probability(stage_idx=stage_idx, total_stages=total_stages)
    p_causal = 1.0 - p_noncausal
    p_roots = p_noncausal * float(p_roots_given_noncausal)
    p_head = p_noncausal * (1.0 - float(p_roots_given_noncausal))
    probs = {"causal": float(p_causal), "head": float(p_head), "roots": float(p_roots)}

    # If an explicit progress is passed with legacy mode, smooth the legacy weights.
    if progress is not None:
        t = float(p**1.0)
        start = {"causal": 0.0, "head": 1.0 - float(p_roots_given_noncausal), "roots": float(p_roots_given_noncausal)}
        for k in probs:
            probs[k] = (1.0 - t) * start[k] + t * probs[k]
        z = float(sum(probs.values()))
        probs = {k: float(v / z) for k, v in probs.items()}

    return probs


def _sample_generation_mode(
    stage_idx: int,
    total_stages: int,
    num_causes: int,
    num_features: int,
    p_roots_given_noncausal: float,
    mode_schedule: Optional[GenerationModeSchedule],
    progress: Optional[float],
    rng: np.random.Generator,
) -> str:
    probs = generation_mode_probabilities(
        stage_idx=stage_idx,
        total_stages=total_stages,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        mode_schedule=mode_schedule,
        progress=progress,
    )

    if int(num_causes) == int(num_features):
        modes = ["causal", "head", "roots"]
    else:
        modes = ["causal", "head"]

    weights = np.array([float(probs.get(m, 0.0)) for m in modes], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.full_like(weights, 1.0 / len(weights))
    else:
        weights = weights / total
    return str(rng.choice(modes, p=weights))


def stage_upper_limit(stage_idx: int, total_stages: int, lo: int, hi: int) -> int:
    """Linear growth of upper bound from lo to hi over stages."""
    p = stage_progress(stage_idx=stage_idx, total_stages=total_stages)
    return _linear_int_upper_from_progress(progress=p, lo=lo, hi=hi)


def stage_linear_probability(stage_idx: int, total_stages: int, start: float, end: float) -> float:
    """Linear schedule from `start` (stage 1) to `end` (stage K)."""
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("start and end must be in [0, 1].")
    p = stage_progress(stage_idx=stage_idx, total_stages=total_stages)
    return float((1.0 - p) * start + p * end)


def smooth_stage_value(
    stage_idx: int,
    total_stages: int,
    lo: int,
    hi: int,
    gamma: float = 1.0,
) -> float:
    """Continuous stage value x(s; K) for smooth curriculum schedules.

    With progress p = (s-1)/(K-1), this returns:
    x = lo + (hi - lo) * p^gamma
    """
    p = stage_progress(stage_idx=stage_idx, total_stages=total_stages)
    schedule = SmoothIntegerSchedule(lo=int(lo), hi=int(hi), gamma=float(gamma))
    return schedule.expected_value(progress=p)


def sample_curriculum_factor_context(
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_causes: int,
    num_features: int,
    p_roots_given_noncausal: float,
    rng: np.random.Generator,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    current_values: Optional[Dict[str, object]] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Dict[str, object]:
    """Sample stage-dependent factors, including optional future factors.

    Returned dict may include keys outside SimplifiedPriorConfig when
    `extra_stage_samplers` provides them.
    """
    p = _resolve_progress(stage_idx=stage_idx, total_stages=total_stages, progress=progress)
    layer_schedule = SmoothIntegerSchedule(
        lo=int(bounds.num_layers_min),
        hi=int(bounds.num_layers_max),
        gamma=float(num_layers_gamma),
    )
    hidden_schedule = SmoothIntegerSchedule(
        lo=int(bounds.hidden_dim_min),
        hi=int(bounds.hidden_dim_max),
        gamma=float(hidden_dim_gamma),
    )

    factor_values: Dict[str, object] = {
        "generation_mode": _sample_generation_mode(
            stage_idx=stage_idx,
            total_stages=total_stages,
            num_causes=int(num_causes),
            num_features=int(num_features),
            p_roots_given_noncausal=float(p_roots_given_noncausal),
            mode_schedule=mode_schedule,
            progress=p,
            rng=rng,
        ),
        "num_layers": int(layer_schedule.sample(progress=p, rng=rng)),
        "hidden_dim": int(hidden_schedule.sample(progress=p, rng=rng)),
    }

    merged_values = dict(current_values or {})
    merged_values.update(factor_values)

    for key, spec in (extra_stage_samplers or {}).items():
        if key in _DERIVED_MODE_FACTORS:
            raise ValueError(f"'{key}' is derived from generation_mode and cannot be stage-sampled directly.")
        sampled = _sample_from_stage_sampler(
            spec=spec,
            progress=p,
            current_values=merged_values,
            rng=rng,
        )
        factor_values[key] = sampled
        merged_values[key] = sampled

    factor_values["curriculum_progress"] = p
    return factor_values


def sample_stage_dependent_hyperparameters(
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_causes: int,
    num_features: int,
    p_roots_given_noncausal: float,
    rng: np.random.Generator,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Dict[str, object]:
    """Sample stage-dependent config fields."""
    factors = sample_curriculum_factor_context(
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        num_causes=num_causes,
        num_features=num_features,
        p_roots_given_noncausal=p_roots_given_noncausal,
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    return {k: v for k, v in factors.items() if k in STAGE_DEPENDENT_FACTORS}


def _split_cfg_and_extra_factors(
    cfg_dict: Dict[str, object],
    factor_context: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    cfg_overrides: Dict[str, object] = {}
    extra_factors: Dict[str, object] = {}
    for key, value in factor_context.items():
        if key == "curriculum_progress":
            continue
        if key in cfg_dict:
            cfg_overrides[key] = value
        else:
            extra_factors[key] = value
    return cfg_overrides, extra_factors


def sample_curriculum_config_with_context(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]]:
    """Sample one curriculum config and return full stage factor context."""
    if rng is None:
        rng = np.random.default_rng()

    cfg_dict = sample_stage_invariant_hyperparameters(
        base_cfg=base_cfg,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    factor_context = sample_curriculum_factor_context(
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        num_causes=int(cfg_dict["num_causes"]),
        num_features=int(cfg_dict["num_features"]),
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        current_values=dict(cfg_dict),
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    cfg_overrides, extra_factors = _split_cfg_and_extra_factors(cfg_dict=cfg_dict, factor_context=factor_context)

    cfg_dict["difficulty"] = None
    cfg_dict.update(cfg_overrides)
    stage_cfg = SimplifiedPriorConfig(**cfg_dict)
    factor_context = dict(factor_context)
    factor_context["curriculum_progress"] = factor_context.get(
        "curriculum_progress",
        _resolve_progress(stage_idx=stage_idx, total_stages=total_stages, progress=progress),
    )
    factor_context["extra_factors"] = extra_factors
    return stage_cfg, factor_context


def sample_curriculum_config(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> SimplifiedPriorConfig:
    """Backward-compatible config sampler (without extra-factor return)."""
    stage_cfg, _ = sample_curriculum_config_with_context(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=p_roots_given_noncausal,
        stationary_sampler=stationary_sampler,
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    return stage_cfg


def sample_smooth_curriculum_config_with_context(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]]:
    """Smooth curriculum wrapper using continuous stage progression.

    Let K be total stages and s be current stage:
    - p(s;K) = (s-1)/(K-1)
    - generation_mode probabilities are annealed via GenerationModeSchedule
    - integer factors use smooth expectation + stochastic rounding:
      x = lo + (hi-lo) * p^gamma

    This yields small expected changes between stages when K is large.
    """
    schedule = mode_schedule or GenerationModeSchedule()
    return sample_curriculum_config_with_context(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        stationary_sampler=stationary_sampler,
        rng=rng,
        mode_schedule=schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )


def sample_smooth_curriculum_config(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> SimplifiedPriorConfig:
    """Smooth curriculum config sampler."""
    stage_cfg, _ = sample_smooth_curriculum_config_with_context(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        stationary_sampler=stationary_sampler,
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    return stage_cfg


def generate_curriculum_stage_batch(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_datasets: int,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    return_context: bool = False,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]] | Tuple[
    SimplifiedPriorConfig,
    Dict[str, object],
    Dict[str, object],
]:
    """Sample a stage config, generate datasets, and optionally return extra factors."""
    stage_cfg, factor_context = sample_curriculum_config_with_context(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        stationary_sampler=stationary_sampler,
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    batch = generate_simplified_prior_data(stage_cfg, num_datasets=num_datasets)
    if return_context:
        return stage_cfg, batch, factor_context
    return stage_cfg, batch


def generate_smooth_curriculum_stage_batch(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_datasets: int,
    p_roots_given_noncausal: float = 0.5,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
    mode_schedule: Optional[GenerationModeSchedule] = None,
    extra_stage_samplers: Optional[Mapping[str, object]] = None,
    progress: Optional[float] = None,
    return_context: bool = False,
    num_layers_gamma: float = 1.0,
    hidden_dim_gamma: float = 1.0,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]] | Tuple[
    SimplifiedPriorConfig,
    Dict[str, object],
    Dict[str, object],
]:
    """Generate a batch with the smooth curriculum sampler."""
    stage_cfg, factor_context = sample_smooth_curriculum_config_with_context(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        p_roots_given_noncausal=float(p_roots_given_noncausal),
        stationary_sampler=stationary_sampler,
        rng=rng,
        mode_schedule=mode_schedule,
        extra_stage_samplers=extra_stage_samplers,
        progress=progress,
        num_layers_gamma=float(num_layers_gamma),
        hidden_dim_gamma=float(hidden_dim_gamma),
    )
    batch = generate_simplified_prior_data(stage_cfg, num_datasets=num_datasets)
    if return_context:
        return stage_cfg, batch, factor_context
    return stage_cfg, batch
