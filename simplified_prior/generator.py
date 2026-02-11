"""Simplified MLP-SCM prior for continuous-target survival pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

GENERATION_MODES: Tuple[str, ...] = ("causal", "head", "roots")
SAMPLING_MODES: Tuple[str, ...] = ("normal", "uniform")


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

    # Randomness / device
    seed: Optional[int] = None
    device: str = "cpu"

    # Optional one-knob difficulty preset.
    difficulty: Optional[str] = None

    def __post_init__(self) -> None:
        self._apply_difficulty_preset()
        self._resolve_and_validate_generation_mode()
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
        if str(self.sampling).lower() not in SAMPLING_MODES:
            raise ValueError("sampling must be one of: 'normal', 'uniform'.")
        if len(self.nonlinearities) == 0:
            raise ValueError("nonlinearities must be non-empty.")

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
        y = _standardize_clip_vector(score).float()
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

    for _ in range(int(num_datasets)):
        prior = SimpleMLPSCMPrior(cfg)
        with torch.no_grad():
            X, y = prior()
        X_list.append(X.detach())
        y_list.append(y.detach())

    X_batch = torch.stack(X_list, dim=0).cpu()
    y_batch = torch.stack(y_list, dim=0).cpu()
    train_sizes = torch.full((int(num_datasets),), train_size, dtype=torch.long)
    seq_lens = torch.full((int(num_datasets),), int(cfg.seq_len), dtype=torch.long)

    return {
        "X": X_batch,
        "y": y_batch,
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
