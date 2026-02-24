"""Simplified MLP-SCM prior for continuous-target survival pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

GENERATION_MODES: Tuple[str, ...] = ("causal", "head", "roots")
SAMPLING_MODES: Tuple[str, ...] = ("normal", "uniform")
TTE_MODELS: Tuple[str, ...] = ("cox", "aft")
COX_BASELINE_TIERS: Tuple[str, ...] = ("tier1", "tier2", "tier3", "tier4")
COX_BASELINE_FAMILIES: Tuple[str, ...] = ("exponential", "weibull", "gompertz", "piecewise", "mixture")
AFT_TIERS: Tuple[str, ...] = ("tier1", "tier2", "tier3", "tier4")
AFT_FAMILIES: Tuple[str, ...] = (
    "normal",
    "logistic",
    "gumbel",
    "student_t",
    "generalized_gamma",
    "gev",
    "skew_normal",
    "mixture",
)
CENSORING_MODES: Tuple[str, ...] = ("log_location", "administrative")
LOG_LOCATION_CENSORING_FAMILIES: Tuple[str, ...] = ("normal", "logistic", "student_t")
ADMINISTRATIVE_JITTER_MODES: Tuple[str, ...] = ("lognormal", "uniform")


class SignActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2.0 * (x >= 0.0).float() - 1.0


class HeavisideActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0.0).float()


class RBFActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-(x**2))


class SineActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class SquareActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x**2


class AbsActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.abs(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "sign": SignActivation,
    "heaviside": HeavisideActivation,
    "rbf": RBFActivation,
    "sine": SineActivation,
    "square": SquareActivation,
    "abs": AbsActivation,
}

_DIFFICULTY_PRESETS = {
    # Presets are defined by the stage-dependent factors.
    "easy": {"generation_mode": "head", "num_layers": 2, "hidden_dim": 16},
    "medium": {"generation_mode": "head", "num_layers": 3, "hidden_dim": 32},
    "hard": {"generation_mode": "causal", "num_layers": 5, "hidden_dim": 64},
}


def _standardize_clip_columns(x: Tensor, clip_value: float = 20.0) -> Tensor:
    """Standardize each column and clip for numerical stability."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((x - mean) / std).clamp(min=-clip_value, max=clip_value)


def _standardize_clip_vector(x: Tensor, clip_value: float = 20.0) -> Tensor:
    """Standardize a 1D tensor and clip for numerical stability."""
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(1e-6)
    return ((x - mean) / std).clamp(min=-clip_value, max=clip_value)


def _make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


@dataclass
class SimplifiedPriorConfig:
    # Dataset size / split
    seq_len: int = 512
    train_size: float | int = 0.5

    # MLP-SCM structure
    num_features: int = 20
    num_causes: int = 20
    num_layers: int = 3
    hidden_dim: int = 32

    # Generation mode
    # - causal: sample X and y from hidden intermediate variables
    # - head:   X from x_head(h), y from y_head(h)
    # - roots:  X from root causes, y from y_head(h)
    # If set to "auto", mode is inferred from is_causal/noncausal_feature_source.
    generation_mode: str = "auto"  # auto | causal | head | roots

    # Auxiliary mode controls (kept in config surface, canonicalized in __post_init__).
    is_causal: bool = False
    noncausal_feature_source: str = "head"  # head | roots

    # Causal-variable selection behavior
    y_is_effect: bool = True
    in_clique: bool = False
    sort_features: bool = True

    # Nonlinearity family
    nonlinearities: Sequence[str] = (
        "tanh",
        "relu",
        "gelu",
        "identity",
        "sign",
        "heaviside",
        "rbf",
        "sine",
        "square",
        "abs",
    )
    per_layer_activation: bool = False

    # Noise / sampling
    noise_std: float = 0.01
    init_std: float = 0.8
    sampling: str = "normal"  # normal | uniform

    # Survival-signal controls.
    standardize_y: bool = True
    y_clip_value: float = 20.0
    nu: float = 1.0  # fixed as eta = y

    # Step 2: choose TTE mechanism per dataset (cox vs aft).
    # - tte_model="auto": draw Bernoulli with P(cox)=p_cox
    # - tte_model="cox" or "aft": fixed mechanism for all datasets
    tte_model: str = "auto"  # auto | cox | aft
    p_cox: float = 0.5

    # Step 3: independent right-censoring generation.
    # - censoring_mode="auto": choose administrative vs log-location
    # - output uses observed_T=min(T, C), delta=1[T <= C]
    censoring_mode: str = "auto"  # auto | log_location | administrative
    p_administrative_censoring: float = 0.5
    censoring_shape_concentration: float = 2.0

    # Log-location matched censoring:
    # log(C) = median(log(T)) + b + eps, where eps has unit scale.
    censoring_log_location_family: str = "auto"  # auto | normal | logistic | student_t
    censoring_log_location_shift_min: float = -0.8
    censoring_log_location_shift_max: float = 0.8
    censoring_log_location_student_df_min: float = 4.0
    censoring_log_location_student_df_max: float = 20.0

    # Administrative censoring:
    # tau = quantile(T, 1 - pi), C = tau * V where median(V)=1.
    censoring_admin_target_rate_min: float = 0.2
    censoring_admin_target_rate_max: float = 0.6
    censoring_admin_jitter_mode: str = "lognormal"  # lognormal | uniform
    censoring_admin_lognormal_sigma: float = 0.15
    censoring_admin_uniform_radius: float = 1.2

    # Guardrails and absolute clipping for censoring times.
    censoring_apply_guardrails: bool = True
    censoring_clamp_min_multiplier: float = 0.5
    censoring_clamp_max_multiplier: float = 2.0
    censoring_time_min: float = 1e-8
    censoring_time_max: float = 1e8

    # Cox baseline sampler controls.
    # - cox_tier="auto": draw from cox_tier_probabilities
    # - cox_tier in {"tier1","tier2","tier3","tier4"}: fixed tier
    cox_tier: str = "auto"  # auto | tier1 | tier2 | tier3 | tier4
    cox_tier_probabilities: Tuple[float, float, float, float] = (0.55, 0.25, 0.15, 0.05)

    # Weibull hyperparameters: k = exp(theta), theta in [-theta_max, theta_max]
    # sampled by symmetric Beta concentration.
    cox_weibull_theta_max: float = 0.8
    cox_weibull_shape_concentration: float = 2.0

    # Gompertz hyperparameters: alpha in [-alpha_max, alpha_max], alpha_max from
    # hazard-ratio cap at reference time: alpha_max = log(hr_max) / ref_time.
    cox_gompertz_hr_max: float = 3.0
    cox_gompertz_reference_time: float = 5.0
    cox_gompertz_shape_concentration: float = 2.0

    # Piecewise baseline hyperparameters.
    cox_piecewise_min_intervals: int = 3
    cox_piecewise_max_intervals: int = 8
    cox_piecewise_t_max: float = 5.0
    cox_piecewise_breakpoint_alpha: float = 2.0
    cox_piecewise_min_width_fraction: float = 0.03
    cox_piecewise_b1_max: float = 0.9
    cox_piecewise_b2_max: float = 0.7
    cox_piecewise_b3_max: float = 0.5
    cox_piecewise_shape_concentration: float = 2.0

    # Cox mixture hyperparameters (tier4).
    cox_mixture_min_components: int = 2
    cox_mixture_max_components: int = 3
    cox_mixture_dirichlet_alpha: float = 2.0
    cox_mixture_component_families: Sequence[str] = ("exponential", "weibull", "gompertz")

    # AFT sampler controls.
    aft_tier: str = "auto"  # auto | tier1 | tier2 | tier3 | tier4
    aft_tier_probabilities: Tuple[float, float, float, float] = (0.55, 0.25, 0.15, 0.05)
    aft_shape_concentration: float = 2.0

    # AFT parameter bounds.
    aft_sigma_min: float = 0.6
    aft_sigma_max: float = 1.8
    aft_student_df_min: float = 3.0
    aft_student_df_max: float = 25.0
    aft_gg_k_min: float = 0.6
    aft_gg_k_max: float = 2.0
    aft_gg_p_min: float = 0.6
    aft_gg_p_max: float = 2.0
    aft_gev_xi_max: float = 0.3
    aft_skew_alpha_max: float = 6.0

    # AFT mixture hyperparameters (tier4).
    aft_mixture_min_components: int = 2
    aft_mixture_max_components: int = 3
    aft_mixture_dirichlet_alpha: float = 2.0
    aft_mixture_component_families: Sequence[str] = (
        "normal",
        "logistic",
        "gumbel",
        "student_t",
        "skew_normal",
    )

    # Randomness / device
    seed: Optional[int] = None
    device: str = "cpu"

    # Optional one-knob difficulty preset.
    difficulty: Optional[str] = None

    def __post_init__(self) -> None:
        self._apply_difficulty_preset()
        self._resolve_and_validate_generation_mode()
        self._resolve_and_validate_tte_model()
        self._resolve_and_validate_censoring_mode()
        self._resolve_and_validate_cox_tier()
        self._resolve_and_validate_aft_tier()
        self._fix_nu()
        self._validate_basic_constraints()
        self._ensure_causal_capacity()

    def _apply_difficulty_preset(self) -> None:
        if self.difficulty is None:
            return
        key = str(self.difficulty).lower()
        if key not in _DIFFICULTY_PRESETS:
            raise ValueError(
                f"Unknown difficulty '{self.difficulty}'. "
                f"Choose from: {sorted(_DIFFICULTY_PRESETS.keys())}."
            )
        preset = _DIFFICULTY_PRESETS[key]
        self.generation_mode = str(preset["generation_mode"])
        self.num_layers = int(preset["num_layers"])
        self.hidden_dim = int(preset["hidden_dim"])

    def _resolve_and_validate_generation_mode(self) -> None:
        source = str(self.noncausal_feature_source).strip().lower()
        if source not in {"head", "roots"}:
            raise ValueError("noncausal_feature_source must be one of: 'head', 'roots'.")

        mode = str(self.generation_mode).strip().lower()
        if mode == "auto":
            mode = "causal" if bool(self.is_causal) else source
        if mode not in GENERATION_MODES:
            raise ValueError("generation_mode must be one of: 'causal', 'head', 'roots', 'auto'.")

        self.generation_mode = mode
        self.is_causal = mode == "causal"
        self.noncausal_feature_source = "roots" if mode == "roots" else "head"

        if mode == "roots" and int(self.num_causes) != int(self.num_features):
            raise ValueError("generation_mode='roots' requires num_causes == num_features.")

    def _validate_basic_constraints(self) -> None:
        if int(self.seq_len) <= 1:
            raise ValueError("seq_len must be > 1.")
        if int(self.num_layers) < 2:
            raise ValueError("num_layers must be >= 2.")
        if int(self.num_features) < 1 or int(self.num_causes) < 1 or int(self.hidden_dim) < 1:
            raise ValueError("num_features, num_causes, and hidden_dim must be >= 1.")
        if float(self.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0.")
        if float(self.init_std) <= 0.0:
            raise ValueError("init_std must be > 0.")
        if float(self.y_clip_value) <= 0.0:
            raise ValueError("y_clip_value must be > 0.")
        if not (0.0 <= float(self.p_cox) <= 1.0):
            raise ValueError("p_cox must be in [0, 1].")
        if not (0.0 <= float(self.p_administrative_censoring) <= 1.0):
            raise ValueError("p_administrative_censoring must be in [0, 1].")
        if float(self.censoring_shape_concentration) <= 0.0:
            raise ValueError("censoring_shape_concentration must be > 0.")
        if float(self.censoring_log_location_shift_max) < float(self.censoring_log_location_shift_min):
            raise ValueError("censoring_log_location_shift_max must be >= censoring_log_location_shift_min.")
        if (
            float(self.censoring_log_location_student_df_min) <= 2.0
            or float(self.censoring_log_location_student_df_max) < float(self.censoring_log_location_student_df_min)
        ):
            raise ValueError("censoring_log_location_student_df bounds must satisfy 2 < min <= max.")
        if (
            float(self.censoring_admin_target_rate_min) < 0.0
            or float(self.censoring_admin_target_rate_max) > 1.0
            or float(self.censoring_admin_target_rate_max) < float(self.censoring_admin_target_rate_min)
        ):
            raise ValueError("censoring_admin_target_rate bounds must satisfy 0 <= min <= max <= 1.")
        if float(self.censoring_admin_lognormal_sigma) < 0.0:
            raise ValueError("censoring_admin_lognormal_sigma must be >= 0.")
        if float(self.censoring_admin_uniform_radius) < 1.0:
            raise ValueError("censoring_admin_uniform_radius must be >= 1.")
        if float(self.censoring_clamp_min_multiplier) <= 0.0:
            raise ValueError("censoring_clamp_min_multiplier must be > 0.")
        if float(self.censoring_clamp_max_multiplier) <= 0.0:
            raise ValueError("censoring_clamp_max_multiplier must be > 0.")
        if float(self.censoring_time_min) <= 0.0:
            raise ValueError("censoring_time_min must be > 0.")
        if float(self.censoring_time_max) < float(self.censoring_time_min):
            raise ValueError("censoring_time_max must be >= censoring_time_min.")
        if len(tuple(self.cox_tier_probabilities)) != 4:
            raise ValueError("cox_tier_probabilities must have length 4 for tiers 1/2/3/4.")
        if float(sum(float(x) for x in self.cox_tier_probabilities)) <= 0.0:
            raise ValueError("cox_tier_probabilities must sum to > 0.")
        if any(float(x) < 0.0 for x in self.cox_tier_probabilities):
            raise ValueError("cox_tier_probabilities must be non-negative.")
        if float(self.cox_weibull_theta_max) < 0.0:
            raise ValueError("cox_weibull_theta_max must be >= 0.")
        if float(self.cox_weibull_shape_concentration) <= 0.0:
            raise ValueError("cox_weibull_shape_concentration must be > 0.")
        if float(self.cox_gompertz_hr_max) < 1.0:
            raise ValueError("cox_gompertz_hr_max must be >= 1.")
        if float(self.cox_gompertz_reference_time) <= 0.0:
            raise ValueError("cox_gompertz_reference_time must be > 0.")
        if float(self.cox_gompertz_shape_concentration) <= 0.0:
            raise ValueError("cox_gompertz_shape_concentration must be > 0.")
        if int(self.cox_piecewise_min_intervals) < 2:
            raise ValueError("cox_piecewise_min_intervals must be >= 2.")
        if int(self.cox_piecewise_max_intervals) < int(self.cox_piecewise_min_intervals):
            raise ValueError("cox_piecewise_max_intervals must be >= cox_piecewise_min_intervals.")
        if float(self.cox_piecewise_t_max) <= 0.0:
            raise ValueError("cox_piecewise_t_max must be > 0.")
        if float(self.cox_piecewise_breakpoint_alpha) <= 0.0:
            raise ValueError("cox_piecewise_breakpoint_alpha must be > 0.")
        if float(self.cox_piecewise_min_width_fraction) < 0.0:
            raise ValueError("cox_piecewise_min_width_fraction must be >= 0.")
        if any(float(x) < 0.0 for x in (self.cox_piecewise_b1_max, self.cox_piecewise_b2_max, self.cox_piecewise_b3_max)):
            raise ValueError("cox piecewise coefficient maxima must be >= 0.")
        if float(self.cox_piecewise_shape_concentration) <= 0.0:
            raise ValueError("cox_piecewise_shape_concentration must be > 0.")
        if int(self.cox_mixture_min_components) < 2:
            raise ValueError("cox_mixture_min_components must be >= 2.")
        if int(self.cox_mixture_max_components) < int(self.cox_mixture_min_components):
            raise ValueError("cox_mixture_max_components must be >= cox_mixture_min_components.")
        if float(self.cox_mixture_dirichlet_alpha) <= 0.0:
            raise ValueError("cox_mixture_dirichlet_alpha must be > 0.")
        if len(tuple(self.cox_mixture_component_families)) == 0:
            raise ValueError("cox_mixture_component_families must be non-empty.")
        allowed_cox_component_families = {"exponential", "weibull", "gompertz"}
        for name in self.cox_mixture_component_families:
            if str(name).strip().lower() not in allowed_cox_component_families:
                raise ValueError(
                    "cox_mixture_component_families must be chosen from "
                    f"{sorted(allowed_cox_component_families)}."
                )
        if len(tuple(self.aft_tier_probabilities)) != 4:
            raise ValueError("aft_tier_probabilities must have length 4 for tiers 1/2/3/4.")
        if float(sum(float(x) for x in self.aft_tier_probabilities)) <= 0.0:
            raise ValueError("aft_tier_probabilities must sum to > 0.")
        if any(float(x) < 0.0 for x in self.aft_tier_probabilities):
            raise ValueError("aft_tier_probabilities must be non-negative.")
        if float(self.aft_shape_concentration) <= 0.0:
            raise ValueError("aft_shape_concentration must be > 0.")
        if float(self.aft_sigma_min) <= 0.0 or float(self.aft_sigma_max) < float(self.aft_sigma_min):
            raise ValueError("aft_sigma bounds must satisfy 0 < min <= max.")
        if float(self.aft_student_df_min) <= 2.0 or float(self.aft_student_df_max) < float(self.aft_student_df_min):
            raise ValueError("aft_student_df bounds must satisfy 2 < min <= max.")
        if float(self.aft_gg_k_min) <= 0.0 or float(self.aft_gg_k_max) < float(self.aft_gg_k_min):
            raise ValueError("aft_gg_k bounds must satisfy 0 < min <= max.")
        if float(self.aft_gg_p_min) <= 0.0 or float(self.aft_gg_p_max) < float(self.aft_gg_p_min):
            raise ValueError("aft_gg_p bounds must satisfy 0 < min <= max.")
        if float(self.aft_gev_xi_max) < 0.0:
            raise ValueError("aft_gev_xi_max must be >= 0.")
        if float(self.aft_skew_alpha_max) < 0.0:
            raise ValueError("aft_skew_alpha_max must be >= 0.")
        if int(self.aft_mixture_min_components) < 2:
            raise ValueError("aft_mixture_min_components must be >= 2.")
        if int(self.aft_mixture_max_components) < int(self.aft_mixture_min_components):
            raise ValueError("aft_mixture_max_components must be >= aft_mixture_min_components.")
        if float(self.aft_mixture_dirichlet_alpha) <= 0.0:
            raise ValueError("aft_mixture_dirichlet_alpha must be > 0.")
        if len(tuple(self.aft_mixture_component_families)) == 0:
            raise ValueError("aft_mixture_component_families must be non-empty.")
        allowed_aft_component_families = {"normal", "logistic", "gumbel", "student_t", "skew_normal"}
        for name in self.aft_mixture_component_families:
            if str(name).strip().lower() not in allowed_aft_component_families:
                raise ValueError(
                    "aft_mixture_component_families must be chosen from "
                    f"{sorted(allowed_aft_component_families)}."
                )
        if str(self.sampling).lower() not in SAMPLING_MODES:
            raise ValueError("sampling must be one of: 'normal', 'uniform'.")
        if len(self.nonlinearities) == 0:
            raise ValueError("nonlinearities must be non-empty.")

    def _resolve_and_validate_tte_model(self) -> None:
        mode = str(self.tte_model).strip().lower()
        if mode not in {"auto", *TTE_MODELS}:
            raise ValueError("tte_model must be one of: 'auto', 'cox', 'aft'.")
        self.tte_model = mode

    def _resolve_and_validate_censoring_mode(self) -> None:
        mode = str(self.censoring_mode).strip().lower()
        if mode not in {"auto", *CENSORING_MODES}:
            raise ValueError("censoring_mode must be one of: 'auto', 'log_location', 'administrative'.")
        self.censoring_mode = mode

        fam = str(self.censoring_log_location_family).strip().lower()
        if fam not in {"auto", *LOG_LOCATION_CENSORING_FAMILIES}:
            raise ValueError(
                "censoring_log_location_family must be one of: "
                "'auto', 'normal', 'logistic', 'student_t'."
            )
        self.censoring_log_location_family = fam

        jitter = str(self.censoring_admin_jitter_mode).strip().lower()
        if jitter not in ADMINISTRATIVE_JITTER_MODES:
            raise ValueError("censoring_admin_jitter_mode must be one of: 'lognormal', 'uniform'.")
        self.censoring_admin_jitter_mode = jitter

    def _resolve_and_validate_cox_tier(self) -> None:
        tier = str(self.cox_tier).strip().lower()
        if tier not in {"auto", *COX_BASELINE_TIERS}:
            raise ValueError("cox_tier must be one of: 'auto', 'tier1', 'tier2', 'tier3', 'tier4'.")
        self.cox_tier = tier

    def _resolve_and_validate_aft_tier(self) -> None:
        tier = str(self.aft_tier).strip().lower()
        if tier not in {"auto", *AFT_TIERS}:
            raise ValueError("aft_tier must be one of: 'auto', 'tier1', 'tier2', 'tier3', 'tier4'.")
        self.aft_tier = tier

    def _fix_nu(self) -> None:
        # nu is fixed to 1 in this project, so eta equals y.
        self.nu = 1.0

    def _ensure_causal_capacity(self) -> None:
        if self.generation_mode != "causal":
            return
        needed = int(self.num_features) + 1
        blocks = max(int(self.num_layers) - 1, 1)
        min_hidden_dim = int(np.ceil(needed / blocks))
        if int(self.hidden_dim) < min_hidden_dim:
            self.hidden_dim = min_hidden_dim

    def resolve_train_size(self) -> int:
        if isinstance(self.train_size, float):
            if not (0.0 < float(self.train_size) < 1.0):
                raise ValueError("If train_size is float, it must be in (0, 1).")
            t = int(int(self.seq_len) * float(self.train_size))
        else:
            t = int(self.train_size)
        if not (0 < t < int(self.seq_len)):
            raise ValueError("Resolved train size must satisfy 0 < train_size < seq_len.")
        return t


class SimpleMLPSCMPrior(nn.Module):
    """MLP-SCM prior that generates (X, y) with continuous y."""

    def __init__(self, cfg: SimplifiedPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.input_layer = nn.Linear(cfg.num_causes, cfg.hidden_dim)
        self.blocks = nn.ModuleList(self._build_hidden_blocks())
        self.x_head = nn.Linear(cfg.hidden_dim, cfg.num_features)
        self.y_head = nn.Linear(cfg.hidden_dim, 1)

        self.to(self.device)
        self._initialize_weights(std=float(cfg.init_std))

    def _build_hidden_blocks(self) -> list[nn.Module]:
        cfg = self.cfg
        block_count = int(cfg.num_layers) - 1
        blocks: list[nn.Module] = []

        if cfg.per_layer_activation:
            activation_names = [str(np.random.choice(list(cfg.nonlinearities))) for _ in range(block_count)]
        else:
            activation_names = [str(cfg.nonlinearities[0])] * block_count

        for act_name in activation_names:
            blocks.append(
                nn.Sequential(
                    _make_activation(act_name),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
            )
        return blocks

    def _initialize_weights(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _sample_root_causes(self) -> Tensor:
        cfg = self.cfg
        if cfg.sampling == "normal":
            return torch.randn(cfg.seq_len, cfg.num_causes, device=self.device)
        if cfg.sampling == "uniform":
            return torch.rand(cfg.seq_len, cfg.num_causes, device=self.device)
        raise ValueError("sampling must be one of: 'normal', 'uniform'.")

    def forward(self) -> Tuple[Tensor, Tensor]:
        causes = self._sample_root_causes()
        h = self.input_layer(causes)

        causal_intermediates: list[Tensor] = []
        collect_intermediates = self.cfg.generation_mode == "causal"

        for block in self.blocks:
            h = block(h)
            if self.cfg.noise_std > 0.0:
                h = h + torch.randn_like(h) * float(self.cfg.noise_std)
            if collect_intermediates:
                causal_intermediates.append(h)

        if self.cfg.generation_mode == "causal":
            X_raw, score = self._sample_causal_X_and_score(causal_intermediates)
        elif self.cfg.generation_mode == "head":
            X_raw = self.x_head(h)
            score = self.y_head(h).squeeze(-1)
        elif self.cfg.generation_mode == "roots":
            X_raw = causes
            score = self.y_head(h).squeeze(-1)
        else:
            raise RuntimeError(f"Unexpected generation mode: {self.cfg.generation_mode}")

        X = _standardize_clip_columns(X_raw).float()
        if bool(self.cfg.standardize_y):
            y = _standardize_clip_vector(score, clip_value=float(self.cfg.y_clip_value)).float()
        else:
            y = score.float()
        return X, y

    def _sample_causal_X_and_score(self, intermediates: list[Tensor]) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        if len(intermediates) == 0:
            raise ValueError("generation_mode='causal' requires at least one hidden block output.")

        pool = torch.cat(intermediates, dim=1)
        total_vars = int(pool.shape[1])
        needed = int(cfg.num_features) + 1
        if total_vars < needed:
            raise ValueError(f"Not enough intermediate variables: have {total_vars}, need {needed}.")

        y_idx = self._sample_target_index(pool)
        x_indices = self._sample_feature_indices(total_vars=total_vars, y_idx=y_idx)

        if cfg.sort_features:
            x_indices, _ = torch.sort(x_indices)

        X = pool[:, x_indices]
        y_score = pool[:, y_idx]
        return X, y_score

    def _sample_target_index(self, pool: Tensor) -> int:
        cfg = self.cfg
        total_vars = int(pool.shape[1])
        block_width = int(cfg.hidden_dim)
        num_blocks = len(self.blocks)

        first_block = torch.arange(0, min(block_width, total_vars), device=pool.device)
        last_start = max(0, (num_blocks - 1) * block_width)
        last_block = torch.arange(last_start, min(last_start + block_width, total_vars), device=pool.device)

        if cfg.y_is_effect and len(last_block) > 0:
            pool_indices = last_block
        elif len(first_block) > 0:
            pool_indices = first_block
        else:
            pool_indices = torch.arange(0, total_vars, device=pool.device)

        picked = torch.randint(0, len(pool_indices), (1,), device=pool.device)
        return int(pool_indices[picked].item())

    def _sample_feature_indices(self, total_vars: int, y_idx: int) -> Tensor:
        cfg = self.cfg
        device = self.device
        feature_count = int(cfg.num_features)

        if cfg.in_clique:
            clique_size = feature_count + 1
            start_min = max(0, y_idx - (clique_size - 1))
            start_max = min(y_idx, total_vars - clique_size)
            start = int(torch.randint(start_min, start_max + 1, (1,), device=device).item())
            clique = torch.arange(start, start + clique_size, device=device)
            return clique[clique != y_idx]

        candidates = torch.arange(0, total_vars, device=device)
        candidates = candidates[candidates != y_idx]
        permuted = candidates[torch.randperm(len(candidates), device=device)]
        return permuted[:feature_count]


def generate_simplified_prior_data(
    cfg: SimplifiedPriorConfig,
    num_datasets: int = 1,
) -> Dict[str, Tensor]:
    """Generate a batch of independent datasets from the simplified prior.

    Returned tensor shapes:
    - X: (B, T, F)
    - y: (B, T)
    - T: (B, T), latent event times
    - log_T: (B, T), latent log event times
    - C: (B, T), censoring times
    - log_C: (B, T), log censoring times
    - observed_T: (B, T), min(T, C)
    - log_observed_T: (B, T)
    - delta: (B, T), event indicators in {0,1} where 1 means uncensored
    - event_indicators: (B, T), boolean alias of delta
    - censoring_mode_ids: (B,), 0=log_location, 1=administrative
    - censoring_rate: (B,), realized fraction censored in each dataset
    - event_rate: (B,), realized fraction uncensored in each dataset
    - censoring_target_rate: (B,), sampled target rate for administrative mode (NaN otherwise)
    - censoring_log_location_shift: (B,), sampled shift b for log-location mode (NaN otherwise)
    - censoring_log_location_family_ids: (B,), -1 when not log-location
    - eta: (B, T), where eta = y (nu fixed to 1)
    - tte_model_ids: (B,), 0=cox and 1=aft
    - tte_is_cox: (B,), boolean indicator
    - cox_tier_ids: (B,), -1 for non-cox
    - cox_family_ids: (B,), -1 for non-cox
    - cox_weibull_k: (B,), NaN if family is not weibull
    - cox_gompertz_alpha: (B,), NaN if family is not gompertz
    - cox_piecewise_num_intervals: (B,), 0 if family is not piecewise
    - cox_piecewise_breakpoints: (B, max_intervals-1), NaN-padded
    - cox_piecewise_hazards: (B, max_intervals), NaN-padded
    - cox_piecewise_b1/b2/b3: (B,), NaN if family is not piecewise
    - cox_mixture_num_components: (B,), 0 if family is not mixture
    - cox_mixture_weights: (B, cox_mixture_max_components), NaN-padded
    - cox_mixture_component_family_ids: (B, cox_mixture_max_components), -1 padded
    - cox_mixture_component_weibull_k/gompertz_alpha: (B, cox_mixture_max_components), NaN-padded
    - aft_tier_ids: (B,), -1 for non-aft
    - aft_family_ids: (B,), -1 for non-aft
    - aft_sigma/student_df/gg_k/gg_p/gev_xi/skew_alpha: (B,), NaN when not applicable
    - aft_mixture_num_components: (B,), 0 if family is not mixture
    - aft_mixture_weights: (B, aft_mixture_max_components), NaN-padded
    - aft_mixture_component_family_ids: (B, aft_mixture_max_components), -1 padded
    - aft_mixture_component_sigma/student_df/skew_alpha: (B, aft_mixture_max_components), NaN-padded
    - train_sizes: (B,)
    - seq_lens: (B,)
    """
    if int(num_datasets) < 1:
        raise ValueError("num_datasets must be >= 1.")

    if cfg.seed is not None:
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))

    train_size = cfg.resolve_train_size()
    X_list: list[Tensor] = []
    y_list: list[Tensor] = []
    t_list: list[Tensor] = []
    log_t_list: list[Tensor] = []
    c_list: list[Tensor] = []
    log_c_list: list[Tensor] = []
    observed_t_list: list[Tensor] = []
    log_observed_t_list: list[Tensor] = []
    delta_list: list[Tensor] = []
    censoring_mode_ids: list[int] = []
    censoring_rates: list[float] = []
    event_rates: list[float] = []
    censoring_target_rates: list[float] = []
    censoring_log_location_shifts: list[float] = []
    censoring_log_location_family_ids: list[int] = []
    tte_model_ids: list[int] = []
    cox_tier_ids: list[int] = []
    cox_family_ids: list[int] = []
    cox_weibull_k: list[float] = []
    cox_gompertz_alpha: list[float] = []
    cox_piecewise_num_intervals: list[int] = []
    cox_piecewise_b1: list[float] = []
    cox_piecewise_b2: list[float] = []
    cox_piecewise_b3: list[float] = []
    cox_mixture_num_components: list[int] = []
    cox_mixture_max_components = int(cfg.cox_mixture_max_components)
    cox_mixture_weights = np.full((int(num_datasets), cox_mixture_max_components), np.nan, dtype=np.float32)
    cox_mixture_component_family_ids = np.full((int(num_datasets), cox_mixture_max_components), -1, dtype=np.int64)
    cox_mixture_component_weibull_k = np.full((int(num_datasets), cox_mixture_max_components), np.nan, dtype=np.float32)
    cox_mixture_component_gompertz_alpha = np.full((int(num_datasets), cox_mixture_max_components), np.nan, dtype=np.float32)

    aft_tier_ids: list[int] = []
    aft_family_ids: list[int] = []
    aft_sigma: list[float] = []
    aft_student_df: list[float] = []
    aft_gg_k: list[float] = []
    aft_gg_p: list[float] = []
    aft_gev_xi: list[float] = []
    aft_skew_alpha: list[float] = []
    aft_mixture_num_components: list[int] = []
    aft_mixture_max_components = int(cfg.aft_mixture_max_components)
    aft_mixture_weights = np.full((int(num_datasets), aft_mixture_max_components), np.nan, dtype=np.float32)
    aft_mixture_component_family_ids = np.full((int(num_datasets), aft_mixture_max_components), -1, dtype=np.int64)
    aft_mixture_component_sigma = np.full((int(num_datasets), aft_mixture_max_components), np.nan, dtype=np.float32)
    aft_mixture_component_student_df = np.full((int(num_datasets), aft_mixture_max_components), np.nan, dtype=np.float32)
    aft_mixture_component_skew_alpha = np.full((int(num_datasets), aft_mixture_max_components), np.nan, dtype=np.float32)

    max_piecewise_intervals = int(cfg.cox_piecewise_max_intervals)
    cox_piecewise_breakpoints = np.full((int(num_datasets), max(max_piecewise_intervals - 1, 0)), np.nan, dtype=np.float32)
    cox_piecewise_hazards = np.full((int(num_datasets), max_piecewise_intervals), np.nan, dtype=np.float32)

    for ds_idx in range(int(num_datasets)):
        sampled_tte_model = sample_tte_model(cfg=cfg)
        tte_model_ids.append(tte_model_to_id(sampled_tte_model))
        sampled_cox_spec: Optional[Dict[str, object]] = None
        sampled_aft_spec: Optional[Dict[str, object]] = None

        if sampled_tte_model == "cox":
            baseline = sample_cox_baseline(cfg=cfg)
            sampled_cox_spec = baseline
            cox_tier_ids.append(cox_tier_to_id(str(baseline["tier"])))
            cox_family_ids.append(cox_baseline_family_to_id(str(baseline["family"])))
            cox_weibull_k.append(float(baseline.get("weibull_k", np.nan)))
            cox_gompertz_alpha.append(float(baseline.get("gompertz_alpha", np.nan)))
            cox_piecewise_num_intervals.append(int(baseline.get("piecewise_num_intervals", 0)))
            cox_piecewise_b1.append(float(baseline.get("piecewise_b1", np.nan)))
            cox_piecewise_b2.append(float(baseline.get("piecewise_b2", np.nan)))
            cox_piecewise_b3.append(float(baseline.get("piecewise_b3", np.nan)))
            cox_mixture_num_components.append(int(baseline.get("mixture_num_components", 0)))

            pw_breaks = baseline.get("piecewise_breakpoints")
            pw_hazards = baseline.get("piecewise_hazards")
            if isinstance(pw_breaks, np.ndarray) and pw_breaks.size > 0:
                take = min(pw_breaks.size, cox_piecewise_breakpoints.shape[1])
                cox_piecewise_breakpoints[ds_idx, :take] = pw_breaks[:take]
            if isinstance(pw_hazards, np.ndarray) and pw_hazards.size > 0:
                take = min(pw_hazards.size, cox_piecewise_hazards.shape[1])
                cox_piecewise_hazards[ds_idx, :take] = pw_hazards[:take]

            mix_w = baseline.get("mixture_weights")
            mix_fam = baseline.get("mixture_component_family_ids")
            mix_wk = baseline.get("mixture_component_weibull_k")
            mix_ga = baseline.get("mixture_component_gompertz_alpha")
            if isinstance(mix_w, np.ndarray) and mix_w.size > 0:
                take = min(mix_w.size, cox_mixture_weights.shape[1])
                cox_mixture_weights[ds_idx, :take] = mix_w[:take]
            if isinstance(mix_fam, np.ndarray) and mix_fam.size > 0:
                take = min(mix_fam.size, cox_mixture_component_family_ids.shape[1])
                cox_mixture_component_family_ids[ds_idx, :take] = mix_fam[:take]
            if isinstance(mix_wk, np.ndarray) and mix_wk.size > 0:
                take = min(mix_wk.size, cox_mixture_component_weibull_k.shape[1])
                cox_mixture_component_weibull_k[ds_idx, :take] = mix_wk[:take]
            if isinstance(mix_ga, np.ndarray) and mix_ga.size > 0:
                take = min(mix_ga.size, cox_mixture_component_gompertz_alpha.shape[1])
                cox_mixture_component_gompertz_alpha[ds_idx, :take] = mix_ga[:take]

            aft_tier_ids.append(-1)
            aft_family_ids.append(-1)
            aft_sigma.append(float("nan"))
            aft_student_df.append(float("nan"))
            aft_gg_k.append(float("nan"))
            aft_gg_p.append(float("nan"))
            aft_gev_xi.append(float("nan"))
            aft_skew_alpha.append(float("nan"))
            aft_mixture_num_components.append(0)
        else:
            aft = sample_aft_spec(cfg=cfg)
            sampled_aft_spec = aft
            aft_tier_ids.append(aft_tier_to_id(str(aft["tier"])))
            aft_family_ids.append(aft_family_to_id(str(aft["family"])))
            aft_sigma.append(float(aft.get("sigma", np.nan)))
            aft_student_df.append(float(aft.get("student_df", np.nan)))
            aft_gg_k.append(float(aft.get("gg_k", np.nan)))
            aft_gg_p.append(float(aft.get("gg_p", np.nan)))
            aft_gev_xi.append(float(aft.get("gev_xi", np.nan)))
            aft_skew_alpha.append(float(aft.get("skew_alpha", np.nan)))
            aft_mixture_num_components.append(int(aft.get("mixture_num_components", 0)))

            cox_tier_ids.append(-1)
            cox_family_ids.append(-1)
            cox_weibull_k.append(float("nan"))
            cox_gompertz_alpha.append(float("nan"))
            cox_piecewise_num_intervals.append(0)
            cox_piecewise_b1.append(float("nan"))
            cox_piecewise_b2.append(float("nan"))
            cox_piecewise_b3.append(float("nan"))
            cox_mixture_num_components.append(0)

            aft_mix_w = aft.get("mixture_weights")
            aft_mix_fam = aft.get("mixture_component_family_ids")
            aft_mix_sigma = aft.get("mixture_component_sigma")
            aft_mix_df = aft.get("mixture_component_student_df")
            aft_mix_skew = aft.get("mixture_component_skew_alpha")
            if isinstance(aft_mix_w, np.ndarray) and aft_mix_w.size > 0:
                take = min(aft_mix_w.size, aft_mixture_weights.shape[1])
                aft_mixture_weights[ds_idx, :take] = aft_mix_w[:take]
            if isinstance(aft_mix_fam, np.ndarray) and aft_mix_fam.size > 0:
                take = min(aft_mix_fam.size, aft_mixture_component_family_ids.shape[1])
                aft_mixture_component_family_ids[ds_idx, :take] = aft_mix_fam[:take]
            if isinstance(aft_mix_sigma, np.ndarray) and aft_mix_sigma.size > 0:
                take = min(aft_mix_sigma.size, aft_mixture_component_sigma.shape[1])
                aft_mixture_component_sigma[ds_idx, :take] = aft_mix_sigma[:take]
            if isinstance(aft_mix_df, np.ndarray) and aft_mix_df.size > 0:
                take = min(aft_mix_df.size, aft_mixture_component_student_df.shape[1])
                aft_mixture_component_student_df[ds_idx, :take] = aft_mix_df[:take]
            if isinstance(aft_mix_skew, np.ndarray) and aft_mix_skew.size > 0:
                take = min(aft_mix_skew.size, aft_mixture_component_skew_alpha.shape[1])
                aft_mixture_component_skew_alpha[ds_idx, :take] = aft_mix_skew[:take]

        prior = SimpleMLPSCMPrior(cfg)
        with torch.no_grad():
            X, y = prior()
        X_list.append(X.detach())
        y_list.append(y.detach())

        eta_np = y_to_linear_predictor(y.detach(), nu=float(cfg.nu)).cpu().numpy().astype(np.float64)
        if sampled_tte_model == "cox":
            if sampled_cox_spec is None:
                raise RuntimeError("Cox spec is missing while tte_model='cox'.")
            t_np, log_t_np = sample_event_times_cox(
                eta=eta_np,
                cox_spec=sampled_cox_spec,
            )
        else:
            if sampled_aft_spec is None:
                raise RuntimeError("AFT spec is missing while tte_model='aft'.")
            t_np, log_t_np = sample_event_times_aft(
                eta=eta_np,
                aft_spec=sampled_aft_spec,
                cfg=cfg,
            )

        censoring_out = sample_right_censoring(
            event_times=t_np,
            cfg=cfg,
        )

        t_list.append(torch.from_numpy(t_np).float())
        log_t_list.append(torch.from_numpy(log_t_np).float())
        c_list.append(torch.from_numpy(np.asarray(censoring_out["C"], dtype=np.float32)).float())
        log_c_list.append(torch.from_numpy(np.asarray(censoring_out["log_C"], dtype=np.float32)).float())
        observed_t_list.append(torch.from_numpy(np.asarray(censoring_out["observed_T"], dtype=np.float32)).float())
        log_observed_t_list.append(torch.from_numpy(np.asarray(censoring_out["log_observed_T"], dtype=np.float32)).float())
        delta_list.append(torch.from_numpy(np.asarray(censoring_out["delta"], dtype=np.float32)).float())
        censoring_mode_ids.append(int(censoring_out["mode_id"]))
        censoring_rates.append(float(censoring_out["censoring_rate"]))
        event_rates.append(float(censoring_out["event_rate"]))
        censoring_target_rates.append(float(censoring_out["target_rate"]))
        censoring_log_location_shifts.append(float(censoring_out["log_location_shift"]))
        censoring_log_location_family_ids.append(int(censoring_out["log_location_family_id"]))

    X_batch = torch.stack(X_list, dim=0).cpu()
    y_batch = torch.stack(y_list, dim=0).cpu()
    T_batch = torch.stack(t_list, dim=0).cpu()
    log_T_batch = torch.stack(log_t_list, dim=0).cpu()
    C_batch = torch.stack(c_list, dim=0).cpu()
    log_C_batch = torch.stack(log_c_list, dim=0).cpu()
    observed_T_batch = torch.stack(observed_t_list, dim=0).cpu()
    log_observed_T_batch = torch.stack(log_observed_t_list, dim=0).cpu()
    delta_batch = torch.stack(delta_list, dim=0).cpu()
    event_indicators_batch = delta_batch > 0.5
    censoring_mode_ids_batch = torch.tensor(censoring_mode_ids, dtype=torch.long)
    censoring_rate_batch = torch.tensor(censoring_rates, dtype=torch.float32)
    event_rate_batch = torch.tensor(event_rates, dtype=torch.float32)
    censoring_target_rate_batch = torch.tensor(censoring_target_rates, dtype=torch.float32)
    censoring_log_location_shift_batch = torch.tensor(censoring_log_location_shifts, dtype=torch.float32)
    censoring_log_location_family_ids_batch = torch.tensor(censoring_log_location_family_ids, dtype=torch.long)
    eta_batch = y_to_linear_predictor(y_batch, nu=float(cfg.nu)).cpu()
    tte_model_ids_batch = torch.tensor(tte_model_ids, dtype=torch.long)
    tte_is_cox_batch = (tte_model_ids_batch == tte_model_to_id("cox"))
    cox_tier_ids_batch = torch.tensor(cox_tier_ids, dtype=torch.long)
    cox_family_ids_batch = torch.tensor(cox_family_ids, dtype=torch.long)
    cox_weibull_k_batch = torch.tensor(cox_weibull_k, dtype=torch.float32)
    cox_gompertz_alpha_batch = torch.tensor(cox_gompertz_alpha, dtype=torch.float32)
    cox_piecewise_num_intervals_batch = torch.tensor(cox_piecewise_num_intervals, dtype=torch.long)
    cox_piecewise_breakpoints_batch = torch.tensor(cox_piecewise_breakpoints, dtype=torch.float32)
    cox_piecewise_hazards_batch = torch.tensor(cox_piecewise_hazards, dtype=torch.float32)
    cox_piecewise_b1_batch = torch.tensor(cox_piecewise_b1, dtype=torch.float32)
    cox_piecewise_b2_batch = torch.tensor(cox_piecewise_b2, dtype=torch.float32)
    cox_piecewise_b3_batch = torch.tensor(cox_piecewise_b3, dtype=torch.float32)
    cox_mixture_num_components_batch = torch.tensor(cox_mixture_num_components, dtype=torch.long)
    cox_mixture_weights_batch = torch.tensor(cox_mixture_weights, dtype=torch.float32)
    cox_mixture_component_family_ids_batch = torch.tensor(cox_mixture_component_family_ids, dtype=torch.long)
    cox_mixture_component_weibull_k_batch = torch.tensor(cox_mixture_component_weibull_k, dtype=torch.float32)
    cox_mixture_component_gompertz_alpha_batch = torch.tensor(cox_mixture_component_gompertz_alpha, dtype=torch.float32)
    aft_tier_ids_batch = torch.tensor(aft_tier_ids, dtype=torch.long)
    aft_family_ids_batch = torch.tensor(aft_family_ids, dtype=torch.long)
    aft_sigma_batch = torch.tensor(aft_sigma, dtype=torch.float32)
    aft_student_df_batch = torch.tensor(aft_student_df, dtype=torch.float32)
    aft_gg_k_batch = torch.tensor(aft_gg_k, dtype=torch.float32)
    aft_gg_p_batch = torch.tensor(aft_gg_p, dtype=torch.float32)
    aft_gev_xi_batch = torch.tensor(aft_gev_xi, dtype=torch.float32)
    aft_skew_alpha_batch = torch.tensor(aft_skew_alpha, dtype=torch.float32)
    aft_mixture_num_components_batch = torch.tensor(aft_mixture_num_components, dtype=torch.long)
    aft_mixture_weights_batch = torch.tensor(aft_mixture_weights, dtype=torch.float32)
    aft_mixture_component_family_ids_batch = torch.tensor(aft_mixture_component_family_ids, dtype=torch.long)
    aft_mixture_component_sigma_batch = torch.tensor(aft_mixture_component_sigma, dtype=torch.float32)
    aft_mixture_component_student_df_batch = torch.tensor(aft_mixture_component_student_df, dtype=torch.float32)
    aft_mixture_component_skew_alpha_batch = torch.tensor(aft_mixture_component_skew_alpha, dtype=torch.float32)
    train_sizes = torch.full((int(num_datasets),), train_size, dtype=torch.long)
    seq_lens = torch.full((int(num_datasets),), int(cfg.seq_len), dtype=torch.long)

    return {
        "X": X_batch,
        "y": y_batch,
        "T": T_batch,
        "log_T": log_T_batch,
        "C": C_batch,
        "log_C": log_C_batch,
        "observed_T": observed_T_batch,
        "log_observed_T": log_observed_T_batch,
        "delta": delta_batch,
        "event_indicators": event_indicators_batch,
        "censoring_mode_ids": censoring_mode_ids_batch,
        "censoring_rate": censoring_rate_batch,
        "event_rate": event_rate_batch,
        "censoring_target_rate": censoring_target_rate_batch,
        "censoring_log_location_shift": censoring_log_location_shift_batch,
        "censoring_log_location_family_ids": censoring_log_location_family_ids_batch,
        "eta": eta_batch,
        "tte_model_ids": tte_model_ids_batch,
        "tte_is_cox": tte_is_cox_batch,
        "cox_tier_ids": cox_tier_ids_batch,
        "cox_family_ids": cox_family_ids_batch,
        "cox_weibull_k": cox_weibull_k_batch,
        "cox_gompertz_alpha": cox_gompertz_alpha_batch,
        "cox_piecewise_num_intervals": cox_piecewise_num_intervals_batch,
        "cox_piecewise_breakpoints": cox_piecewise_breakpoints_batch,
        "cox_piecewise_hazards": cox_piecewise_hazards_batch,
        "cox_piecewise_b1": cox_piecewise_b1_batch,
        "cox_piecewise_b2": cox_piecewise_b2_batch,
        "cox_piecewise_b3": cox_piecewise_b3_batch,
        "cox_mixture_num_components": cox_mixture_num_components_batch,
        "cox_mixture_weights": cox_mixture_weights_batch,
        "cox_mixture_component_family_ids": cox_mixture_component_family_ids_batch,
        "cox_mixture_component_weibull_k": cox_mixture_component_weibull_k_batch,
        "cox_mixture_component_gompertz_alpha": cox_mixture_component_gompertz_alpha_batch,
        "aft_tier_ids": aft_tier_ids_batch,
        "aft_family_ids": aft_family_ids_batch,
        "aft_sigma": aft_sigma_batch,
        "aft_student_df": aft_student_df_batch,
        "aft_gg_k": aft_gg_k_batch,
        "aft_gg_p": aft_gg_p_batch,
        "aft_gev_xi": aft_gev_xi_batch,
        "aft_skew_alpha": aft_skew_alpha_batch,
        "aft_mixture_num_components": aft_mixture_num_components_batch,
        "aft_mixture_weights": aft_mixture_weights_batch,
        "aft_mixture_component_family_ids": aft_mixture_component_family_ids_batch,
        "aft_mixture_component_sigma": aft_mixture_component_sigma_batch,
        "aft_mixture_component_student_df": aft_mixture_component_student_df_batch,
        "aft_mixture_component_skew_alpha": aft_mixture_component_skew_alpha_batch,
        "train_sizes": train_sizes,
        "seq_lens": seq_lens,
    }


def split_dataset(X: Tensor, y: Tensor, train_size: int) -> Dict[str, Tensor]:
    """Split a single dataset into train/test segments."""
    t = int(train_size)
    return {
        "X_train": X[:t],
        "y_train": y[:t],
        "X_test": X[t:],
        "y_test": y[t:],
    }


def available_nonlinearities() -> Iterable[str]:
    return tuple(sorted(_ACTIVATIONS.keys()))


def available_difficulties() -> Iterable[str]:
    return tuple(sorted(_DIFFICULTY_PRESETS.keys()))


def available_tte_models() -> Iterable[str]:
    return TTE_MODELS


def available_censoring_modes() -> Iterable[str]:
    return CENSORING_MODES


def available_log_location_censoring_families() -> Iterable[str]:
    return LOG_LOCATION_CENSORING_FAMILIES


def available_cox_baseline_tiers() -> Iterable[str]:
    return COX_BASELINE_TIERS


def available_cox_baseline_families() -> Iterable[str]:
    return COX_BASELINE_FAMILIES


def available_aft_tiers() -> Iterable[str]:
    return AFT_TIERS


def available_aft_families() -> Iterable[str]:
    return AFT_FAMILIES


def y_to_linear_predictor(y: Tensor, nu: float = 1.0) -> Tensor:
    """Convert scalar y to linear predictor eta.

    In this project nu is fixed to 1, so eta = y.
    """
    _ = nu  # kept for API compatibility
    return y.float()


def tte_model_to_id(model: str) -> int:
    """Map TTE model name to integer ID (cox=0, aft=1)."""
    key = str(model).strip().lower()
    if key == "cox":
        return 0
    if key == "aft":
        return 1
    raise ValueError("Unknown TTE model. Choose from: 'cox', 'aft'.")


def tte_model_from_id(model_id: int) -> str:
    """Map integer ID to TTE model name (0=cox, 1=aft)."""
    idx = int(model_id)
    if idx == 0:
        return "cox"
    if idx == 1:
        return "aft"
    raise ValueError("Unknown TTE model ID. Expected 0 or 1.")


def censoring_mode_to_id(mode: str) -> int:
    key = str(mode).strip().lower()
    if key == "log_location":
        return 0
    if key == "administrative":
        return 1
    raise ValueError("Unknown censoring mode. Choose from: 'log_location', 'administrative'.")


def censoring_mode_from_id(mode_id: int) -> str:
    idx = int(mode_id)
    if idx == 0:
        return "log_location"
    if idx == 1:
        return "administrative"
    raise ValueError("Unknown censoring mode ID. Expected 0 or 1.")


def log_location_censoring_family_to_id(family: str) -> int:
    key = str(family).strip().lower()
    if key == "normal":
        return 0
    if key == "logistic":
        return 1
    if key == "student_t":
        return 2
    raise ValueError("Unknown log-location censoring family. Choose from: 'normal', 'logistic', 'student_t'.")


def log_location_censoring_family_from_id(family_id: int) -> str:
    idx = int(family_id)
    if idx == 0:
        return "normal"
    if idx == 1:
        return "logistic"
    if idx == 2:
        return "student_t"
    raise ValueError("Unknown log-location censoring family ID. Expected 0, 1, or 2.")


def sample_censoring_mode(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> str:
    """Sample dataset-level censoring mode according to config."""
    if cfg.censoring_mode in CENSORING_MODES:
        return str(cfg.censoring_mode)

    if rng is None:
        draw = float(np.random.random())
    else:
        draw = float(rng.random())
    if draw < float(cfg.p_administrative_censoring):
        return "administrative"
    return "log_location"


def sample_log_location_censoring_family(
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Sample log-location censoring family (or return fixed family)."""
    if cfg.censoring_log_location_family in LOG_LOCATION_CENSORING_FAMILIES:
        return str(cfg.censoring_log_location_family)
    return _sample_choice(
        values=list(LOG_LOCATION_CENSORING_FAMILIES),
        probs=[0.45, 0.35, 0.20],
        rng=rng,
    )


def sample_tte_model(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> str:
    """Sample the dataset-level TTE model according to cfg.tte_model/p_cox."""
    if cfg.tte_model in TTE_MODELS:
        return str(cfg.tte_model)

    if rng is None:
        draw = float(np.random.random())
    else:
        draw = float(rng.random())
    return "cox" if draw < float(cfg.p_cox) else "aft"


def cox_tier_to_id(tier: str) -> int:
    key = str(tier).strip().lower()
    if key == "tier1":
        return 0
    if key == "tier2":
        return 1
    if key == "tier3":
        return 2
    if key == "tier4":
        return 3
    raise ValueError("Unknown Cox tier. Choose from: 'tier1', 'tier2', 'tier3', 'tier4'.")


def cox_tier_from_id(tier_id: int) -> str:
    idx = int(tier_id)
    if idx == 0:
        return "tier1"
    if idx == 1:
        return "tier2"
    if idx == 2:
        return "tier3"
    if idx == 3:
        return "tier4"
    raise ValueError("Unknown Cox tier ID. Expected 0, 1, 2, or 3.")


def cox_baseline_family_to_id(family: str) -> int:
    key = str(family).strip().lower()
    if key == "exponential":
        return 0
    if key == "weibull":
        return 1
    if key == "gompertz":
        return 2
    if key == "piecewise":
        return 3
    if key == "mixture":
        return 4
    raise ValueError("Unknown Cox baseline family.")


def cox_baseline_family_from_id(family_id: int) -> str:
    idx = int(family_id)
    if idx == 0:
        return "exponential"
    if idx == 1:
        return "weibull"
    if idx == 2:
        return "gompertz"
    if idx == 3:
        return "piecewise"
    if idx == 4:
        return "mixture"
    raise ValueError("Unknown Cox baseline family ID. Expected 0..4.")


def aft_tier_to_id(tier: str) -> int:
    key = str(tier).strip().lower()
    if key == "tier1":
        return 0
    if key == "tier2":
        return 1
    if key == "tier3":
        return 2
    if key == "tier4":
        return 3
    raise ValueError("Unknown AFT tier. Choose from: 'tier1', 'tier2', 'tier3', 'tier4'.")


def aft_tier_from_id(tier_id: int) -> str:
    idx = int(tier_id)
    if idx == 0:
        return "tier1"
    if idx == 1:
        return "tier2"
    if idx == 2:
        return "tier3"
    if idx == 3:
        return "tier4"
    raise ValueError("Unknown AFT tier ID. Expected 0, 1, 2, or 3.")


def aft_family_to_id(family: str) -> int:
    key = str(family).strip().lower()
    mapping = {
        "normal": 0,
        "logistic": 1,
        "gumbel": 2,
        "student_t": 3,
        "generalized_gamma": 4,
        "gev": 5,
        "skew_normal": 6,
        "mixture": 7,
    }
    if key not in mapping:
        raise ValueError("Unknown AFT family.")
    return int(mapping[key])


def aft_family_from_id(family_id: int) -> str:
    idx = int(family_id)
    mapping = {
        0: "normal",
        1: "logistic",
        2: "gumbel",
        3: "student_t",
        4: "generalized_gamma",
        5: "gev",
        6: "skew_normal",
        7: "mixture",
    }
    if idx not in mapping:
        raise ValueError("Unknown AFT family ID. Expected 0..7.")
    return mapping[idx]


def _sample_choice(
    values: Sequence[str],
    probs: Sequence[float],
    rng: Optional[np.random.Generator] = None,
) -> str:
    p = np.array([float(x) for x in probs], dtype=np.float64)
    total = float(p.sum())
    if total <= 0.0:
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / total
    if rng is None:
        idx = int(np.random.choice(len(values), p=p))
    else:
        idx = int(rng.choice(len(values), p=p))
    return str(values[idx])


def _sample_symmetric_beta(
    max_abs: float,
    concentration: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    m = float(max_abs)
    if m <= 0.0:
        return 0.0
    a = float(concentration)
    if rng is None:
        u = float(np.random.beta(a, a))
    else:
        u = float(rng.beta(a, a))
    return m * (2.0 * u - 1.0)


def _sample_piecewise_interval_count(
    cfg: SimplifiedPriorConfig,
    tier: str,
    rng: Optional[np.random.Generator] = None,
) -> int:
    lo = int(cfg.cox_piecewise_min_intervals)
    hi = int(cfg.cox_piecewise_max_intervals)
    if lo == hi:
        return lo

    mid = int((lo + hi) // 2)
    if tier == "tier2":
        lo_t, hi_t = lo, max(lo, mid)
    elif tier == "tier3":
        lo_t, hi_t = min(hi, max(lo, mid)), hi
    else:
        lo_t, hi_t = lo, hi

    if rng is None:
        return int(np.random.randint(lo_t, hi_t + 1))
    return int(rng.integers(lo_t, hi_t + 1))


def _sample_piecewise_breakpoints(
    cfg: SimplifiedPriorConfig,
    n_intervals: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = np.full((n_intervals,), float(cfg.cox_piecewise_breakpoint_alpha), dtype=np.float64)
    if rng is None:
        widths = np.random.dirichlet(alpha)
    else:
        widths = rng.dirichlet(alpha)

    min_frac = float(cfg.cox_piecewise_min_width_fraction)
    max_allowed = (1.0 - 1e-8) / max(n_intervals, 1)
    if min_frac > max_allowed:
        min_frac = max_allowed
    if min_frac > 0.0:
        widths = (1.0 - min_frac * n_intervals) * widths + min_frac
        widths = widths / np.sum(widths)

    t_max = float(cfg.cox_piecewise_t_max)
    cumulative = np.cumsum(widths) * t_max
    breakpoints = cumulative[:-1].astype(np.float32)
    return breakpoints, widths.astype(np.float32)


def _sample_piecewise_hazards(
    cfg: SimplifiedPriorConfig,
    widths: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, float, float]:
    t_max = float(cfg.cox_piecewise_t_max)
    edges = np.concatenate(([0.0], np.cumsum(widths) * t_max))
    mids = ((edges[:-1] + edges[1:]) / 2.0) / max(t_max, 1e-8)

    conc = float(cfg.cox_piecewise_shape_concentration)
    b1 = _sample_symmetric_beta(float(cfg.cox_piecewise_b1_max), conc, rng=rng)
    b2 = _sample_symmetric_beta(float(cfg.cox_piecewise_b2_max), conc, rng=rng)
    b3 = _sample_symmetric_beta(float(cfg.cox_piecewise_b3_max), conc, rng=rng)

    log_lambda = b1 * mids + b2 * (mids**2 - (1.0 / 3.0)) + b3 * np.sin(2.0 * np.pi * mids)
    hazards = np.exp(log_lambda).astype(np.float64)

    mean_hazard = float(np.sum(hazards * widths))
    hazards = hazards / max(mean_hazard, 1e-8)
    return hazards.astype(np.float32), float(b1), float(b2), float(b3)


def _sample_weibull_k(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> float:
    theta = _sample_symmetric_beta(
        max_abs=float(cfg.cox_weibull_theta_max),
        concentration=float(cfg.cox_weibull_shape_concentration),
        rng=rng,
    )
    return float(np.exp(theta))


def _sample_gompertz_alpha(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> float:
    alpha_max = float(np.log(float(cfg.cox_gompertz_hr_max)) / float(cfg.cox_gompertz_reference_time))
    return _sample_symmetric_beta(
        max_abs=alpha_max,
        concentration=float(cfg.cox_gompertz_shape_concentration),
        rng=rng,
    )


def _sample_beta_range(
    low: float,
    high: float,
    concentration: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if float(high) <= float(low):
        return float(low)
    a = float(concentration)
    if rng is None:
        u = float(np.random.beta(a, a))
    else:
        u = float(rng.beta(a, a))
    return float(low + (high - low) * u)


def _sample_cox_component_family(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> str:
    values = [str(x).strip().lower() for x in cfg.cox_mixture_component_families]
    probs = np.full((len(values),), 1.0 / max(len(values), 1), dtype=np.float64)
    return _sample_choice(values=values, probs=probs, rng=rng)


def _sample_cox_non_mixture_params(
    family: str,
    cfg: SimplifiedPriorConfig,
    tier: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    spec: Dict[str, object] = {
        "weibull_k": np.nan,
        "gompertz_alpha": np.nan,
        "piecewise_num_intervals": 0,
        "piecewise_breakpoints": np.empty((0,), dtype=np.float32),
        "piecewise_hazards": np.empty((0,), dtype=np.float32),
        "piecewise_b1": np.nan,
        "piecewise_b2": np.nan,
        "piecewise_b3": np.nan,
    }

    if family == "weibull":
        spec["weibull_k"] = _sample_weibull_k(cfg=cfg, rng=rng)
        return spec
    if family == "gompertz":
        spec["gompertz_alpha"] = _sample_gompertz_alpha(cfg=cfg, rng=rng)
        return spec
    if family == "piecewise":
        n_intervals = _sample_piecewise_interval_count(cfg=cfg, tier=tier, rng=rng)
        breakpoints, widths = _sample_piecewise_breakpoints(cfg=cfg, n_intervals=n_intervals, rng=rng)
        hazards, b1, b2, b3 = _sample_piecewise_hazards(cfg=cfg, widths=widths, rng=rng)
        spec["piecewise_num_intervals"] = int(n_intervals)
        spec["piecewise_breakpoints"] = breakpoints
        spec["piecewise_hazards"] = hazards
        spec["piecewise_b1"] = float(b1)
        spec["piecewise_b2"] = float(b2)
        spec["piecewise_b3"] = float(b3)
        return spec
    if family != "exponential":
        raise ValueError(f"Unexpected Cox family: {family}")
    return spec


def sample_cox_baseline(
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample one Cox baseline family and parameters for a dataset."""
    if cfg.cox_tier == "auto":
        tier = _sample_choice(
            values=list(COX_BASELINE_TIERS),
            probs=list(cfg.cox_tier_probabilities),
            rng=rng,
        )
    else:
        tier = str(cfg.cox_tier)

    family_values: Sequence[str]
    family_probs: Sequence[float]
    if tier == "tier1":
        family_values = ("exponential", "weibull", "gompertz")
        family_probs = (0.50, 0.30, 0.20)
    elif tier == "tier2":
        family_values = ("exponential", "weibull", "gompertz", "piecewise")
        family_probs = (0.15, 0.35, 0.25, 0.25)
    elif tier == "tier3":
        family_values = ("weibull", "gompertz", "piecewise")
        family_probs = (0.20, 0.20, 0.60)
    elif tier == "tier4":
        family_values = ("mixture",)
        family_probs = (1.0,)
    else:
        raise ValueError(f"Unexpected Cox tier: {tier}")

    family = _sample_choice(values=family_values, probs=family_probs, rng=rng)
    spec: Dict[str, object] = {
        "tier": tier,
        "family": family,
        "weibull_k": np.nan,
        "gompertz_alpha": np.nan,
        "piecewise_num_intervals": 0,
        "piecewise_breakpoints": np.empty((0,), dtype=np.float32),
        "piecewise_hazards": np.empty((0,), dtype=np.float32),
        "piecewise_b1": np.nan,
        "piecewise_b2": np.nan,
        "piecewise_b3": np.nan,
        "mixture_num_components": 0,
        "mixture_weights": np.empty((0,), dtype=np.float32),
        "mixture_component_family_ids": np.empty((0,), dtype=np.int64),
        "mixture_component_weibull_k": np.empty((0,), dtype=np.float32),
        "mixture_component_gompertz_alpha": np.empty((0,), dtype=np.float32),
    }

    if family != "mixture":
        spec.update(_sample_cox_non_mixture_params(family=family, cfg=cfg, tier=tier, rng=rng))
        return spec

    m_lo = int(cfg.cox_mixture_min_components)
    m_hi = int(cfg.cox_mixture_max_components)
    if rng is None:
        m = int(np.random.randint(m_lo, m_hi + 1))
        weights = np.random.dirichlet(np.full((m,), float(cfg.cox_mixture_dirichlet_alpha), dtype=np.float64))
    else:
        m = int(rng.integers(m_lo, m_hi + 1))
        weights = rng.dirichlet(np.full((m,), float(cfg.cox_mixture_dirichlet_alpha), dtype=np.float64))

    comp_family_ids = np.full((m,), -1, dtype=np.int64)
    comp_weibull_k = np.full((m,), np.nan, dtype=np.float32)
    comp_gompertz_alpha = np.full((m,), np.nan, dtype=np.float32)

    for i in range(m):
        comp_family = _sample_cox_component_family(cfg=cfg, rng=rng)
        comp_family_ids[i] = cox_baseline_family_to_id(comp_family)
        comp_spec = _sample_cox_non_mixture_params(family=comp_family, cfg=cfg, tier="tier3", rng=rng)
        if np.isfinite(float(comp_spec["weibull_k"])):
            comp_weibull_k[i] = float(comp_spec["weibull_k"])
        if np.isfinite(float(comp_spec["gompertz_alpha"])):
            comp_gompertz_alpha[i] = float(comp_spec["gompertz_alpha"])

    spec["mixture_num_components"] = int(m)
    spec["mixture_weights"] = weights.astype(np.float32)
    spec["mixture_component_family_ids"] = comp_family_ids
    spec["mixture_component_weibull_k"] = comp_weibull_k
    spec["mixture_component_gompertz_alpha"] = comp_gompertz_alpha
    return spec


def _sample_aft_base_family_params(
    family: str,
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    params = {
        "sigma": np.nan,
        "student_df": np.nan,
        "gg_k": np.nan,
        "gg_p": np.nan,
        "gev_xi": np.nan,
        "skew_alpha": np.nan,
    }
    conc = float(cfg.aft_shape_concentration)

    if family in {"normal", "logistic", "gumbel", "student_t", "skew_normal"}:
        params["sigma"] = _sample_beta_range(
            low=float(cfg.aft_sigma_min),
            high=float(cfg.aft_sigma_max),
            concentration=conc,
            rng=rng,
        )
    if family == "student_t":
        params["student_df"] = _sample_beta_range(
            low=float(cfg.aft_student_df_min),
            high=float(cfg.aft_student_df_max),
            concentration=conc,
            rng=rng,
        )
    if family == "generalized_gamma":
        params["gg_k"] = _sample_beta_range(
            low=float(cfg.aft_gg_k_min),
            high=float(cfg.aft_gg_k_max),
            concentration=conc,
            rng=rng,
        )
        params["gg_p"] = _sample_beta_range(
            low=float(cfg.aft_gg_p_min),
            high=float(cfg.aft_gg_p_max),
            concentration=conc,
            rng=rng,
        )
    if family == "gev":
        params["gev_xi"] = _sample_symmetric_beta(
            max_abs=float(cfg.aft_gev_xi_max),
            concentration=conc,
            rng=rng,
        )
    if family == "skew_normal":
        params["skew_alpha"] = _sample_symmetric_beta(
            max_abs=float(cfg.aft_skew_alpha_max),
            concentration=conc,
            rng=rng,
        )
    return params


def _sample_aft_component_family(cfg: SimplifiedPriorConfig, rng: Optional[np.random.Generator] = None) -> str:
    values = [str(x).strip().lower() for x in cfg.aft_mixture_component_families]
    probs = np.full((len(values),), 1.0 / max(len(values), 1), dtype=np.float64)
    return _sample_choice(values=values, probs=probs, rng=rng)


def sample_aft_spec(
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample one AFT family and parameters for a dataset."""
    if cfg.aft_tier == "auto":
        tier = _sample_choice(
            values=list(AFT_TIERS),
            probs=list(cfg.aft_tier_probabilities),
            rng=rng,
        )
    else:
        tier = str(cfg.aft_tier)

    if tier == "tier1":
        family_values = ("normal", "logistic", "gumbel")
        family_probs = (0.40, 0.30, 0.30)
    elif tier == "tier2":
        family_values = ("student_t", "generalized_gamma", "gev")
        family_probs = (0.40, 0.35, 0.25)
    elif tier == "tier3":
        family_values = ("skew_normal", "student_t", "generalized_gamma", "gev")
        family_probs = (0.40, 0.25, 0.20, 0.15)
    elif tier == "tier4":
        family_values = ("mixture",)
        family_probs = (1.0,)
    else:
        raise ValueError(f"Unexpected AFT tier: {tier}")

    family = _sample_choice(values=family_values, probs=family_probs, rng=rng)
    params = _sample_aft_base_family_params(family=family, cfg=cfg, rng=rng)
    spec: Dict[str, object] = {
        "tier": tier,
        "family": family,
        "sigma": params["sigma"],
        "student_df": params["student_df"],
        "gg_k": params["gg_k"],
        "gg_p": params["gg_p"],
        "gev_xi": params["gev_xi"],
        "skew_alpha": params["skew_alpha"],
        "mixture_num_components": 0,
        "mixture_weights": np.empty((0,), dtype=np.float32),
        "mixture_component_family_ids": np.empty((0,), dtype=np.int64),
        "mixture_component_sigma": np.empty((0,), dtype=np.float32),
        "mixture_component_student_df": np.empty((0,), dtype=np.float32),
        "mixture_component_skew_alpha": np.empty((0,), dtype=np.float32),
    }

    if family != "mixture":
        return spec

    m_lo = int(cfg.aft_mixture_min_components)
    m_hi = int(cfg.aft_mixture_max_components)
    if rng is None:
        m = int(np.random.randint(m_lo, m_hi + 1))
        weights = np.random.dirichlet(np.full((m,), float(cfg.aft_mixture_dirichlet_alpha), dtype=np.float64))
    else:
        m = int(rng.integers(m_lo, m_hi + 1))
        weights = rng.dirichlet(np.full((m,), float(cfg.aft_mixture_dirichlet_alpha), dtype=np.float64))

    comp_family_ids = np.full((m,), -1, dtype=np.int64)
    comp_sigma = np.full((m,), np.nan, dtype=np.float32)
    comp_student_df = np.full((m,), np.nan, dtype=np.float32)
    comp_skew_alpha = np.full((m,), np.nan, dtype=np.float32)

    for i in range(m):
        comp_family = _sample_aft_component_family(cfg=cfg, rng=rng)
        comp_family_ids[i] = aft_family_to_id(comp_family)
        comp_params = _sample_aft_base_family_params(family=comp_family, cfg=cfg, rng=rng)
        if np.isfinite(float(comp_params["sigma"])):
            comp_sigma[i] = float(comp_params["sigma"])
        if np.isfinite(float(comp_params["student_df"])):
            comp_student_df[i] = float(comp_params["student_df"])
        if np.isfinite(float(comp_params["skew_alpha"])):
            comp_skew_alpha[i] = float(comp_params["skew_alpha"])

    spec["mixture_num_components"] = int(m)
    spec["mixture_weights"] = weights.astype(np.float32)
    spec["mixture_component_family_ids"] = comp_family_ids
    spec["mixture_component_sigma"] = comp_sigma
    spec["mixture_component_student_df"] = comp_student_df
    spec["mixture_component_skew_alpha"] = comp_skew_alpha
    spec["sigma"] = np.nan
    spec["student_df"] = np.nan
    spec["gg_k"] = np.nan
    spec["gg_p"] = np.nan
    spec["gev_xi"] = np.nan
    spec["skew_alpha"] = np.nan
    return spec


def _sample_uniform(shape: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        return np.random.random(size=shape)
    return rng.random(size=shape)


def _sanitize_event_times(
    t: np.ndarray,
    t_min: float = 1e-8,
    t_max: float = 1e8,
) -> Tuple[np.ndarray, np.ndarray]:
    t64 = np.asarray(t, dtype=np.float64)
    t64 = np.nan_to_num(t64, nan=float(t_max), posinf=float(t_max), neginf=float(t_min))
    t64 = np.clip(t64, float(t_min), float(t_max))
    log_t = np.log(t64)
    return t64.astype(np.float32), log_t.astype(np.float32)


def _cox_inverse_gompertz(z: np.ndarray, alpha: float) -> np.ndarray:
    z64 = np.asarray(z, dtype=np.float64)
    a = float(alpha)
    if abs(a) <= 1e-10:
        return z64
    arg = 1.0 + a * z64
    out = np.full_like(z64, np.inf, dtype=np.float64)
    valid = arg > 1e-12
    out[valid] = np.log(arg[valid]) / a
    return out


def _cox_inverse_piecewise(z: np.ndarray, breakpoints: np.ndarray, hazards: np.ndarray) -> np.ndarray:
    z64 = np.asarray(z, dtype=np.float64)
    h = np.maximum(np.asarray(hazards, dtype=np.float64), 1e-8)
    if h.size == 0:
        return np.full_like(z64, np.nan, dtype=np.float64)
    if h.size == 1:
        return z64 / h[0]

    bp = np.sort(np.asarray(breakpoints, dtype=np.float64))
    target = max(h.size - 1, 0)
    if bp.size < target:
        pad_value = float(bp[-1]) if bp.size > 0 else 0.0
        bp = np.concatenate((bp, np.full((target - bp.size,), pad_value, dtype=np.float64)))
    if bp.size > target:
        bp = bp[:target]
    widths = np.diff(np.concatenate(([0.0], bp)))
    widths = np.maximum(widths, 1e-8)
    cum_end = np.cumsum(h[:-1] * widths)

    out = np.empty_like(z64, dtype=np.float64)
    if cum_end.size == 0:
        out[:] = z64 / h[-1]
        return out

    tail_start_h = float(cum_end[-1])
    tail_start_t = float(bp[-1]) if bp.size > 0 else 0.0
    is_tail = z64 >= tail_start_h

    z_fin = z64[~is_tail]
    if z_fin.size > 0:
        idx = np.searchsorted(cum_end, z_fin, side="right")
        prev = np.zeros_like(z_fin)
        has_prev = idx > 0
        prev[has_prev] = cum_end[idx[has_prev] - 1]
        starts = np.concatenate(([0.0], bp))[idx]
        out[~is_tail] = starts + (z_fin - prev) / h[idx]

    out[is_tail] = tail_start_t + (z64[is_tail] - tail_start_h) / h[-1]
    return out


def _cox_inverse_mixture(z: np.ndarray, cox_spec: Dict[str, object], rng: Optional[np.random.Generator] = None) -> np.ndarray:
    z64 = np.asarray(z, dtype=np.float64)
    weights = np.asarray(cox_spec.get("mixture_weights", np.empty((0,), dtype=np.float32)), dtype=np.float64)
    m = int(cox_spec.get("mixture_num_components", weights.size))
    if m <= 0 or weights.size == 0:
        return z64
    weights = weights[:m]
    weights = weights / max(np.sum(weights), 1e-12)

    if rng is None:
        comp_idx = np.random.choice(m, size=z64.shape[0], p=weights)
    else:
        comp_idx = rng.choice(m, size=z64.shape[0], p=weights)

    fam_ids = np.asarray(cox_spec.get("mixture_component_family_ids", np.full((m,), -1)), dtype=np.int64)
    wks = np.asarray(cox_spec.get("mixture_component_weibull_k", np.full((m,), np.nan)), dtype=np.float64)
    gas = np.asarray(cox_spec.get("mixture_component_gompertz_alpha", np.full((m,), np.nan)), dtype=np.float64)

    out = np.empty_like(z64, dtype=np.float64)
    for i in range(m):
        mask = comp_idx == i
        if not np.any(mask):
            continue
        fam = cox_baseline_family_from_id(int(fam_ids[i])) if i < fam_ids.size and fam_ids[i] >= 0 else "exponential"
        if fam == "weibull":
            k = float(wks[i]) if i < wks.size and np.isfinite(wks[i]) else 1.0
            out[mask] = np.power(np.maximum(z64[mask], 0.0), 1.0 / max(k, 1e-8))
        elif fam == "gompertz":
            alpha = float(gas[i]) if i < gas.size and np.isfinite(gas[i]) else 0.0
            out[mask] = _cox_inverse_gompertz(z64[mask], alpha=alpha)
        else:
            out[mask] = z64[mask]
    return out


def sample_event_times_cox(
    eta: np.ndarray,
    cox_spec: Dict[str, object],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample event times from a Cox model given eta and a sampled baseline spec."""
    eta64 = np.asarray(eta, dtype=np.float64)
    eta64 = np.clip(eta64, -20.0, 20.0)
    u = _sample_uniform(shape=eta64.shape, rng=rng)
    z = -np.log(np.clip(u, 1e-12, 1.0)) / np.exp(eta64)

    family = str(cox_spec.get("family", "exponential")).strip().lower()
    if family == "exponential":
        t = z
    elif family == "weibull":
        k = float(cox_spec.get("weibull_k", 1.0))
        t = np.power(np.maximum(z, 0.0), 1.0 / max(k, 1e-8))
    elif family == "gompertz":
        alpha = float(cox_spec.get("gompertz_alpha", 0.0))
        t = _cox_inverse_gompertz(z=z, alpha=alpha)
    elif family == "piecewise":
        t = _cox_inverse_piecewise(
            z=z,
            breakpoints=np.asarray(cox_spec.get("piecewise_breakpoints", np.empty((0,), dtype=np.float32))),
            hazards=np.asarray(cox_spec.get("piecewise_hazards", np.empty((0,), dtype=np.float32))),
        )
    elif family == "mixture":
        t = _cox_inverse_mixture(z=z, cox_spec=cox_spec, rng=rng)
    else:
        raise ValueError(f"Unknown Cox family for sampling: {family}")

    return _sanitize_event_times(t)


def _sample_aft_noise_family(
    family: str,
    size: int,
    sigma: float = np.nan,
    student_df: float = np.nan,
    gg_k: float = np.nan,
    gg_p: float = np.nan,
    gev_xi: float = np.nan,
    skew_alpha: float = np.nan,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    fam = str(family).strip().lower()
    sig = 1.0 if not np.isfinite(sigma) else max(float(sigma), 1e-8)

    if fam == "normal":
        return np.random.normal(loc=0.0, scale=sig, size=size) if rng is None else rng.normal(loc=0.0, scale=sig, size=size)
    if fam == "logistic":
        return np.random.logistic(loc=0.0, scale=sig, size=size) if rng is None else rng.logistic(loc=0.0, scale=sig, size=size)
    if fam == "gumbel":
        return np.random.gumbel(loc=0.0, scale=sig, size=size) if rng is None else rng.gumbel(loc=0.0, scale=sig, size=size)
    if fam == "student_t":
        df = 6.0 if not np.isfinite(student_df) else max(float(student_df), 2.1)
        base = np.random.standard_t(df=df, size=size) if rng is None else rng.standard_t(df=df, size=size)
        return sig * base
    if fam == "generalized_gamma":
        k = 1.0 if not np.isfinite(gg_k) else max(float(gg_k), 1e-8)
        p = 1.0 if not np.isfinite(gg_p) else max(float(gg_p), 1e-8)
        g = np.random.gamma(shape=k, scale=1.0, size=size) if rng is None else rng.gamma(shape=k, scale=1.0, size=size)
        return np.log(np.maximum(g, 1e-12)) / p
    if fam == "gev":
        xi = 0.0 if not np.isfinite(gev_xi) else float(gev_xi)
        u = _sample_uniform(shape=(size,), rng=rng)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        if abs(xi) <= 1e-10:
            return -np.log(-np.log(u))
        return (np.power(-np.log(u), -xi) - 1.0) / xi
    if fam == "skew_normal":
        alpha = 0.0 if not np.isfinite(skew_alpha) else float(skew_alpha)
        delta = alpha / np.sqrt(1.0 + alpha**2)
        if rng is None:
            z0 = np.random.normal(size=size)
            z1 = np.random.normal(size=size)
        else:
            z0 = rng.normal(size=size)
            z1 = rng.normal(size=size)
        raw = delta * np.abs(z0) + np.sqrt(max(1.0 - delta**2, 1e-12)) * z1
        centered = raw - delta * np.sqrt(2.0 / np.pi)
        return sig * centered
    raise ValueError(f"Unknown AFT family for noise sampling: {family}")


def _sample_aft_noise_mixture(
    shape: Tuple[int, ...],
    aft_spec: Dict[str, object],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    n = int(np.prod(shape))
    weights = np.asarray(aft_spec.get("mixture_weights", np.empty((0,), dtype=np.float32)), dtype=np.float64)
    m = int(aft_spec.get("mixture_num_components", weights.size))
    if m <= 0 or weights.size == 0:
        return np.zeros(shape, dtype=np.float64)
    weights = weights[:m]
    weights = weights / max(np.sum(weights), 1e-12)

    if rng is None:
        comp_idx = np.random.choice(m, size=n, p=weights)
    else:
        comp_idx = rng.choice(m, size=n, p=weights)

    fam_ids = np.asarray(aft_spec.get("mixture_component_family_ids", np.full((m,), -1)), dtype=np.int64)
    sigmas = np.asarray(aft_spec.get("mixture_component_sigma", np.full((m,), np.nan)), dtype=np.float64)
    dfs = np.asarray(aft_spec.get("mixture_component_student_df", np.full((m,), np.nan)), dtype=np.float64)
    skews = np.asarray(aft_spec.get("mixture_component_skew_alpha", np.full((m,), np.nan)), dtype=np.float64)

    eps = np.empty((n,), dtype=np.float64)
    for i in range(m):
        mask = comp_idx == i
        if not np.any(mask):
            continue
        fam = aft_family_from_id(int(fam_ids[i])) if i < fam_ids.size and fam_ids[i] >= 0 else "normal"
        eps[mask] = _sample_aft_noise_family(
            family=fam,
            size=int(np.sum(mask)),
            sigma=float(sigmas[i]) if i < sigmas.size else np.nan,
            student_df=float(dfs[i]) if i < dfs.size else np.nan,
            skew_alpha=float(skews[i]) if i < skews.size else np.nan,
            rng=rng,
        )
    return eps.reshape(shape)


def sample_event_times_aft(
    eta: np.ndarray,
    aft_spec: Dict[str, object],
    cfg: Optional[SimplifiedPriorConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample event times from an AFT model given eta and a sampled family spec."""
    eta64 = np.asarray(eta, dtype=np.float64)
    eta64 = np.clip(eta64, -20.0, 20.0)

    family = str(aft_spec.get("family", "normal")).strip().lower()
    if family == "mixture":
        eps = _sample_aft_noise_mixture(shape=eta64.shape, aft_spec=aft_spec, rng=rng)
    else:
        eps = _sample_aft_noise_family(
            family=family,
            size=int(eta64.size),
            sigma=float(aft_spec.get("sigma", np.nan)),
            student_df=float(aft_spec.get("student_df", np.nan)),
            gg_k=float(aft_spec.get("gg_k", np.nan)),
            gg_p=float(aft_spec.get("gg_p", np.nan)),
            gev_xi=float(aft_spec.get("gev_xi", np.nan)),
            skew_alpha=float(aft_spec.get("skew_alpha", np.nan)),
            rng=rng,
        ).reshape(eta64.shape)

    log_t = eta64 + eps
    with np.errstate(over="ignore", invalid="ignore"):
        t = np.exp(log_t)
    return _sanitize_event_times(t)


def _sample_censoring_times_log_location(
    event_times: np.ndarray,
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    t64 = np.asarray(event_times, dtype=np.float64)
    _, log_t = _sanitize_event_times(
        t64,
        t_min=float(cfg.censoring_time_min),
        t_max=float(cfg.censoring_time_max),
    )
    log_t64 = np.asarray(log_t, dtype=np.float64)

    family = sample_log_location_censoring_family(cfg=cfg, rng=rng)
    shift = _sample_beta_range(
        low=float(cfg.censoring_log_location_shift_min),
        high=float(cfg.censoring_log_location_shift_max),
        concentration=float(cfg.censoring_shape_concentration),
        rng=rng,
    )
    mu = float(np.median(log_t64)) + float(shift)

    student_df = np.nan
    if family == "student_t":
        student_df = _sample_beta_range(
            low=float(cfg.censoring_log_location_student_df_min),
            high=float(cfg.censoring_log_location_student_df_max),
            concentration=float(cfg.censoring_shape_concentration),
            rng=rng,
        )

    eps = _sample_aft_noise_family(
        family=family,
        size=int(log_t64.size),
        sigma=1.0,
        student_df=float(student_df),
        rng=rng,
    ).reshape(log_t64.shape)
    log_c = mu + eps
    with np.errstate(over="ignore", invalid="ignore"):
        c = np.exp(log_c)

    return {
        "mode": "log_location",
        "mode_id": censoring_mode_to_id("log_location"),
        "C_raw": c,
        "target_rate": np.nan,
        "log_location_shift": float(shift),
        "log_location_family": family,
        "log_location_family_id": log_location_censoring_family_to_id(family),
    }


def _sample_censoring_times_administrative(
    event_times: np.ndarray,
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    t64 = np.asarray(event_times, dtype=np.float64)
    t64, _ = _sanitize_event_times(
        t64,
        t_min=float(cfg.censoring_time_min),
        t_max=float(cfg.censoring_time_max),
    )
    t64 = np.asarray(t64, dtype=np.float64)

    target_rate = _sample_beta_range(
        low=float(cfg.censoring_admin_target_rate_min),
        high=float(cfg.censoring_admin_target_rate_max),
        concentration=float(cfg.censoring_shape_concentration),
        rng=rng,
    )
    q = float(np.clip(1.0 - float(target_rate), 0.0, 1.0))
    tau = float(np.quantile(t64, q=q))
    tau = max(tau, float(cfg.censoring_time_min))

    jitter_mode = str(cfg.censoring_admin_jitter_mode)
    if jitter_mode == "lognormal":
        sigma = float(cfg.censoring_admin_lognormal_sigma)
        if rng is None:
            jitter = np.exp(np.random.normal(loc=0.0, scale=sigma, size=t64.shape))
        else:
            jitter = np.exp(rng.normal(loc=0.0, scale=sigma, size=t64.shape))
    else:
        radius = float(cfg.censoring_admin_uniform_radius)
        lo = 1.0 / max(radius, 1e-8)
        hi = max(radius, lo)
        if rng is None:
            jitter = np.random.uniform(low=lo, high=hi, size=t64.shape)
        else:
            jitter = rng.uniform(low=lo, high=hi, size=t64.shape)

    c = tau * jitter
    return {
        "mode": "administrative",
        "mode_id": censoring_mode_to_id("administrative"),
        "C_raw": c,
        "target_rate": float(target_rate),
        "log_location_shift": np.nan,
        "log_location_family": "",
        "log_location_family_id": -1,
    }


def _apply_censoring_guardrails(
    c_raw: np.ndarray,
    event_times: np.ndarray,
    cfg: SimplifiedPriorConfig,
) -> np.ndarray:
    c64 = np.asarray(c_raw, dtype=np.float64)
    t64 = np.asarray(event_times, dtype=np.float64)

    if not bool(cfg.censoring_apply_guardrails):
        return c64

    l = float(np.quantile(t64, 0.05))
    u = float(np.quantile(t64, 0.95))
    l = max(l, float(cfg.censoring_time_min))
    u = max(u, l)

    lower = float(cfg.censoring_clamp_min_multiplier) * l
    upper = float(cfg.censoring_clamp_max_multiplier) * u

    lower = max(lower, float(cfg.censoring_time_min))
    upper = min(upper, float(cfg.censoring_time_max))
    if upper < lower:
        upper = lower
    return np.clip(c64, lower, upper)


def sample_right_censoring(
    event_times: np.ndarray,
    cfg: SimplifiedPriorConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample right-censored observations from latent event times.

    Returns:
    - C, log_C: censoring times
    - observed_T, log_observed_T: observed follow-up times
    - delta: event indicator in {0,1}, where 1 means event observed (not censored)
    - mode/mode_id and sampled censoring metadata
    """
    t, _ = _sanitize_event_times(
        np.asarray(event_times, dtype=np.float64),
        t_min=float(cfg.censoring_time_min),
        t_max=float(cfg.censoring_time_max),
    )
    t64 = np.asarray(t, dtype=np.float64)

    mode = sample_censoring_mode(cfg=cfg, rng=rng)
    if mode == "administrative":
        sampled = _sample_censoring_times_administrative(event_times=t64, cfg=cfg, rng=rng)
    else:
        sampled = _sample_censoring_times_log_location(event_times=t64, cfg=cfg, rng=rng)

    c_raw = np.asarray(sampled["C_raw"], dtype=np.float64)
    c_guarded = _apply_censoring_guardrails(c_raw=c_raw, event_times=t64, cfg=cfg)
    c, log_c = _sanitize_event_times(
        c_guarded,
        t_min=float(cfg.censoring_time_min),
        t_max=float(cfg.censoring_time_max),
    )
    c64 = np.asarray(c, dtype=np.float64)

    observed = np.minimum(t64, c64)
    observed, log_observed = _sanitize_event_times(
        observed,
        t_min=float(cfg.censoring_time_min),
        t_max=float(cfg.censoring_time_max),
    )
    delta_bool = t64 <= c64
    delta = delta_bool.astype(np.float32)
    event_rate = float(np.mean(delta))
    censoring_rate = 1.0 - event_rate

    return {
        "mode": str(sampled["mode"]),
        "mode_id": int(sampled["mode_id"]),
        "C": c,
        "log_C": log_c,
        "observed_T": observed,
        "log_observed_T": log_observed,
        "delta": delta,
        "event_rate": event_rate,
        "censoring_rate": censoring_rate,
        "target_rate": float(sampled["target_rate"]),
        "log_location_shift": float(sampled["log_location_shift"]),
        "log_location_family_id": int(sampled["log_location_family_id"]),
    }
