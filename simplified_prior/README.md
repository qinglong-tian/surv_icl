# Simplified Prior Manual

This document is a detailed manual for sampling prior data from `simplified_prior`.

It covers:
- full data-generation pipeline
- exact APIs to call
- all outputs
- every `SimplifiedPriorConfig` parameter (all 80 fields)
- practical recipes

## 1) What this module generates

The main API returns synthetic tabular survival-style datasets.

For each dataset:
- features: `X`
- continuous latent target: `y`
- linear predictor: `eta` (fixed as `eta = y`)
- latent event time: `T`
- censoring time: `C`
- observed time: `observed_T = min(T, C)`
- right-censoring indicator: `delta = 1[T <= C]`

## 2) Core API

### 2.1) Main classes/functions

- `SimplifiedPriorConfig`
- `generate_simplified_prior_data(cfg, num_datasets=1)`
- `split_dataset(X, y, train_size)`

### 2.2) Minimal usage

```python
from simplified_prior import SimplifiedPriorConfig, generate_simplified_prior_data

cfg = SimplifiedPriorConfig(
    seq_len=512,
    num_features=20,
    num_causes=20,
    generation_mode="auto",
    tte_model="auto",
    censoring_mode="auto",
    seed=42,
)

out = generate_simplified_prior_data(cfg, num_datasets=4)

X = out["X"]                    # (B, T, F)
y = out["y"]                    # (B, T)
T_latent = out["T"]             # (B, T)
T_obs = out["observed_T"]       # (B, T)
delta = out["delta"]            # (B, T), 1 event observed, 0 censored
```

### 2.3) Reproducibility

- Set `seed` in `SimplifiedPriorConfig`.
- The module seeds both NumPy and PyTorch internally.

## 3) Pipeline semantics

The generator executes this order per dataset:

1. sample MLP-SCM tabular data `(X, y)`
2. convert to linear predictor `eta = y` (`nu` is fixed to 1)
3. choose TTE mechanism (Cox or AFT)
4. sample latent event times `T`
5. sample censoring times `C`
6. emit right-censored observation `(observed_T, delta)`

### 3.1) Step 1: MLP-SCM tabular generation

`SimpleMLPSCMPrior` generates `X` and `y` using:
- root-cause inputs
- hidden nonlinear blocks
- mode-dependent extraction (`causal`, `head`, `roots`)

### 3.2) Step 2: latent signal

`eta = y`

Note: `nu` exists in config for compatibility but is forced to `1.0` in `__post_init__`.

### 3.3) Step 3: TTE mechanism

- Cox PH: `h(t | X) = h0(t) * exp(eta)`
- AFT: `log(T) = eta + eps`

### 3.4) Step 4: Cox baseline sampling

Cox tier and family are sampled from configured probabilities, then inverse-transform sampling is applied.

Generic Cox sampling step:
- sample `U ~ Uniform(0,1)`
- compute `z = -log(U) / exp(eta)`
- set `T = H0^{-1}(z)` where `H0` is cumulative baseline hazard

`eta` is clipped to `[-20, 20]` before sampling, and resulting times are sanitized to positive finite bounds.

#### 3.4.1) Cox tier-to-family structure

- if `cox_tier=\"auto\"`, tier is sampled using `cox_tier_probabilities`
- family candidates by tier:
  - `tier1`: `exponential` (0.50), `weibull` (0.30), `gompertz` (0.20)
  - `tier2`: `exponential` (0.15), `weibull` (0.35), `gompertz` (0.25), `piecewise` (0.25)
  - `tier3`: `weibull` (0.20), `gompertz` (0.20), `piecewise` (0.60)
  - `tier4`: `mixture` (1.00)

#### 3.4.2) Cox family details

- `exponential`
  - baseline hazard: `h0(t)=1`
  - cumulative hazard: `H0(t)=t`
  - inverse: `T=z`
  - parameters: none

- `weibull`
  - baseline hazard: `h0(t)=k t^{k-1}`
  - cumulative hazard: `H0(t)=t^k`
  - inverse: `T=z^(1/k)`
  - shape parameter sampling:
    - sample `u ~ Beta(a,a)`, `a = cox_weibull_shape_concentration`
    - `theta = cox_weibull_theta_max * (2u-1)`
    - `k = exp(theta)`
    - therefore `k in [exp(-cox_weibull_theta_max), exp(cox_weibull_theta_max)]`

- `gompertz`
  - baseline hazard: `h0(t)=exp(alpha t)`
  - cumulative hazard: `H0(t)=(exp(alpha t)-1)/alpha` (and `H0(t)=t` when `alpha≈0`)
  - inverse:
    - `T = log(1 + alpha z)/alpha` when numerically stable
    - falls back to exponential-like behavior near `alpha=0`
  - shape parameter sampling:
    - `alpha_max = log(cox_gompertz_hr_max) / cox_gompertz_reference_time`
    - sample `alpha` symmetrically in `[-alpha_max, alpha_max]` using concentration `cox_gompertz_shape_concentration`

- `piecewise`
  - baseline hazard: constant per interval
    - `h0(t)=lambda_j` for `t_(j-1) < t <= t_j`
  - interval count sampling:
    - draw `n_intervals` in `[cox_piecewise_min_intervals, cox_piecewise_max_intervals]`
    - tier-aware bias:
      - `tier2`: lower half of interval range
      - `tier3`: upper half of interval range
  - breakpoint sampling:
    - sample interval widths from Dirichlet with concentration `cox_piecewise_breakpoint_alpha`
    - enforce minimum interval fraction `cox_piecewise_min_width_fraction`
    - scale cumulative widths to `[0, cox_piecewise_t_max]`
  - hazard-shape sampling:
    - normalized interval midpoints `m` in `[0,1]`
    - sample coefficients:
      - `b1` in `[-cox_piecewise_b1_max, cox_piecewise_b1_max]`
      - `b2` in `[-cox_piecewise_b2_max, cox_piecewise_b2_max]`
      - `b3` in `[-cox_piecewise_b3_max, cox_piecewise_b3_max]`
      - all via symmetric-beta concentration `cox_piecewise_shape_concentration`
    - construct:
      - `log(lambda)=b1*m + b2*(m^2 - 1/3) + b3*sin(2*pi*m)`
      - `lambda = exp(log(lambda))`
    - normalize hazards to weighted mean 1 across interval widths
  - inverse:
    - piecewise inversion over cumulative hazard segments plus tail segment

- `mixture`
  - samples a finite mixture of Cox baselines (component-level heterogeneity)
  - component count:
    - `m ~ UniformInt[cox_mixture_min_components, cox_mixture_max_components]`
  - weights:
    - `w ~ Dirichlet(alpha)` with `alpha = cox_mixture_dirichlet_alpha`
  - component family pool:
    - from `cox_mixture_component_families` (allowed: `exponential`, `weibull`, `gompertz`)
  - each component parameterized using same rules as non-mixture family samplers
  - sampling:
    - for each row, choose component index from categorical `w`
    - apply that component's inverse transform

### 3.5) Step 5: AFT family sampling

AFT uses:
- `log(T) = eta + eps`
- `T = exp(log(T))`

`eta` is clipped to `[-20,20]` before noise is added, then output times are sanitized to positive finite bounds.

#### 3.5.1) AFT tier-to-family structure

- if `aft_tier=\"auto\"`, tier is sampled using `aft_tier_probabilities`
- family candidates by tier:
  - `tier1`: `normal` (0.40), `logistic` (0.30), `gumbel` (0.30)
  - `tier2`: `student_t` (0.40), `generalized_gamma` (0.35), `gev` (0.25)
  - `tier3`: `skew_normal` (0.40), `student_t` (0.25), `generalized_gamma` (0.20), `gev` (0.15)
  - `tier4`: `mixture` (1.00)

#### 3.5.2) AFT family details

- `normal`
  - `eps ~ Normal(0, sigma)`
  - `sigma` sampled in `[aft_sigma_min, aft_sigma_max]`

- `logistic`
  - `eps ~ Logistic(0, sigma)`
  - `sigma` sampled in `[aft_sigma_min, aft_sigma_max]`

- `gumbel`
  - `eps ~ Gumbel(0, sigma)`
  - `sigma` sampled in `[aft_sigma_min, aft_sigma_max]`

- `student_t`
  - `eps = sigma * t_df`
  - `sigma` sampled in `[aft_sigma_min, aft_sigma_max]`
  - `df` sampled in `[aft_student_df_min, aft_student_df_max]`

- `generalized_gamma`
  - sample `G ~ Gamma(k, 1)`
  - set `eps = log(G)/p`
  - `k` sampled in `[aft_gg_k_min, aft_gg_k_max]`
  - `p` sampled in `[aft_gg_p_min, aft_gg_p_max]`

- `gev`
  - sample `U ~ Uniform(0,1)`
  - if `xi≈0`: `eps = -log(-log(U))`
  - else: `eps = ((-log(U))^(-xi)-1)/xi`
  - `xi` sampled symmetrically in `[-aft_gev_xi_max, aft_gev_xi_max]`

- `skew_normal`
  - with skew parameter `alpha`:
    - `delta = alpha/sqrt(1+alpha^2)`
    - `z0,z1 ~ Normal(0,1)`
    - `raw = delta*|z0| + sqrt(1-delta^2)*z1`
    - centered to zero mean, then scaled by `sigma`
  - `sigma` sampled in `[aft_sigma_min, aft_sigma_max]`
  - `alpha` sampled symmetrically in `[-aft_skew_alpha_max, aft_skew_alpha_max]`

- `mixture`
  - component count:
    - `m ~ UniformInt[aft_mixture_min_components, aft_mixture_max_components]`
  - weights:
    - `w ~ Dirichlet(alpha)` with `alpha = aft_mixture_dirichlet_alpha`
  - component family pool:
    - from `aft_mixture_component_families`
    - allowed default set: `normal`, `logistic`, `gumbel`, `student_t`, `skew_normal`
  - component parameters sampled by the same family-specific rules
  - for each row, a component is chosen from categorical `w` and its `eps` sampler is applied

#### 3.5.3) Shared parameter sampling helper

For bounded parameters (`sigma`, `df`, `k`, `p`, shifts/rates, etc.), code uses beta-range sampling:
- sample `u ~ Beta(a,a)`
- map to interval `[L, U]` via `L + (U-L)*u`
- concentration `a` controls centrality vs edge mass (higher `a` -> more centered).

### 3.6) Step 6: right censoring sampling

Two censoring modes:

- `administrative`
  - sample target censoring rate `pi`
  - set `tau = quantile(T, 1 - pi)`
  - sample jitter `V` (lognormal or uniform with median near 1)
  - set `C = tau * V`

- `log_location`
  - compute `m = median(log T)`
  - sample shift `b`
  - sample `eps` from chosen family (`normal/logistic/student_t`) with unit scale
  - set `log C = m + b + eps`, `C = exp(log C)`

Guardrails optionally clamp censoring times using robust `T` quantiles:
- `L = Q_T(0.05)`, `U = Q_T(0.95)`
- clamp to `[c_min * L, c_max * U]`

## 4) Output dictionary reference

`generate_simplified_prior_data` returns a dictionary with 50 keys.

### 4.1) Core tensors

| Key | Shape | Meaning |
|---|---:|---|
| `X` | `(B, T, F)` | features |
| `y` | `(B, T)` | continuous scalar target |
| `eta` | `(B, T)` | linear predictor, equal to `y` |
| `T` | `(B, T)` | latent event times |
| `log_T` | `(B, T)` | latent log event times |
| `C` | `(B, T)` | censoring times |
| `log_C` | `(B, T)` | log censoring times |
| `observed_T` | `(B, T)` | observed times `min(T, C)` |
| `log_observed_T` | `(B, T)` | log observed times |
| `delta` | `(B, T)` | `1` if event observed, `0` if censored |
| `event_indicators` | `(B, T)` bool | boolean alias of `delta` |
| `train_sizes` | `(B,)` | resolved train split size |
| `seq_lens` | `(B,)` | sequence length |

### 4.2) TTE/censoring metadata

| Key | Shape | Meaning |
|---|---:|---|
| `tte_model_ids` | `(B,)` | `0=cox`, `1=aft` |
| `tte_is_cox` | `(B,)` bool | model indicator |
| `censoring_mode_ids` | `(B,)` | `0=log_location`, `1=administrative` |
| `censoring_rate` | `(B,)` | realized censored fraction |
| `event_rate` | `(B,)` | realized uncensored fraction |
| `censoring_target_rate` | `(B,)` | admin target rate, `NaN` otherwise |
| `censoring_log_location_shift` | `(B,)` | log-location shift `b`, `NaN` otherwise |
| `censoring_log_location_family_ids` | `(B,)` | `normal=0`, `logistic=1`, `student_t=2`, `-1` otherwise |

### 4.3) Cox metadata

| Key | Shape | Meaning |
|---|---:|---|
| `cox_tier_ids` | `(B,)` | tier id or `-1` for non-Cox |
| `cox_family_ids` | `(B,)` | family id or `-1` for non-Cox |
| `cox_weibull_k` | `(B,)` | Weibull `k` or `NaN` |
| `cox_gompertz_alpha` | `(B,)` | Gompertz `alpha` or `NaN` |
| `cox_piecewise_num_intervals` | `(B,)` | intervals count or `0` |
| `cox_piecewise_breakpoints` | `(B, max_intervals-1)` | padded breakpoints |
| `cox_piecewise_hazards` | `(B, max_intervals)` | padded interval hazards |
| `cox_piecewise_b1` | `(B,)` | piecewise coefficient |
| `cox_piecewise_b2` | `(B,)` | piecewise coefficient |
| `cox_piecewise_b3` | `(B,)` | piecewise coefficient |
| `cox_mixture_num_components` | `(B,)` | mixture components or `0` |
| `cox_mixture_weights` | `(B, cox_mixture_max_components)` | padded weights |
| `cox_mixture_component_family_ids` | `(B, cox_mixture_max_components)` | padded family ids |
| `cox_mixture_component_weibull_k` | `(B, cox_mixture_max_components)` | padded |
| `cox_mixture_component_gompertz_alpha` | `(B, cox_mixture_max_components)` | padded |

### 4.4) AFT metadata

| Key | Shape | Meaning |
|---|---:|---|
| `aft_tier_ids` | `(B,)` | tier id or `-1` for non-AFT |
| `aft_family_ids` | `(B,)` | family id or `-1` for non-AFT |
| `aft_sigma` | `(B,)` | sigma or `NaN` |
| `aft_student_df` | `(B,)` | t df or `NaN` |
| `aft_gg_k` | `(B,)` | generalized gamma `k` or `NaN` |
| `aft_gg_p` | `(B,)` | generalized gamma `p` or `NaN` |
| `aft_gev_xi` | `(B,)` | GEV xi or `NaN` |
| `aft_skew_alpha` | `(B,)` | skew-normal alpha or `NaN` |
| `aft_mixture_num_components` | `(B,)` | mixture components or `0` |
| `aft_mixture_weights` | `(B, aft_mixture_max_components)` | padded weights |
| `aft_mixture_component_family_ids` | `(B, aft_mixture_max_components)` | padded family ids |
| `aft_mixture_component_sigma` | `(B, aft_mixture_max_components)` | padded |
| `aft_mixture_component_student_df` | `(B, aft_mixture_max_components)` | padded |
| `aft_mixture_component_skew_alpha` | `(B, aft_mixture_max_components)` | padded |

## 5) Full parameter reference (`SimplifiedPriorConfig`)

All 80 fields are listed below.

### 5.1) Dataset and split

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `seq_len` | `int` | `512` | must be `> 1` |
| `train_size` | `float | int` | `0.5` | float must be `(0,1)`; resolved train size must satisfy `0 < train_size < seq_len` |

### 5.2) MLP-SCM structure

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `num_features` | `int` | `20` | must be `>= 1` |
| `num_causes` | `int` | `20` | must be `>= 1`; if `generation_mode='roots'` then must equal `num_features` |
| `num_layers` | `int` | `3` | must be `>= 2` |
| `hidden_dim` | `int` | `32` | must be `>= 1`; auto-adjusted upward in causal mode if needed |

### 5.3) Generation mode and causal selection

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `generation_mode` | `str` | `"auto"` | `auto | causal | head | roots` |
| `is_causal` | `bool` | `False` | only used when `generation_mode='auto'` |
| `noncausal_feature_source` | `str` | `"head"` | `head | roots`, used for auto mode |
| `y_is_effect` | `bool` | `True` | controls target index sampling region in causal mode |
| `in_clique` | `bool` | `False` | if true, feature indices are sampled from a clique around target |
| `sort_features` | `bool` | `True` | sort selected feature indices |

### 5.4) Nonlinearity, noise, sampling

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `nonlinearities` | `Sequence[str]` | `(tanh, relu, gelu, identity, sign, heaviside, rbf, sine, square, abs)` | non-empty list of activation names |
| `per_layer_activation` | `bool` | `False` | if false, first nonlinearity is used for all layers |
| `noise_std` | `float` | `0.01` | must be `>= 0` |
| `init_std` | `float` | `0.8` | must be `> 0` |
| `sampling` | `str` | `"normal"` | `normal | uniform` root-cause sampling |

### 5.5) Latent target and signal

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `standardize_y` | `bool` | `True` | standardize/clamp `y` |
| `y_clip_value` | `float` | `20.0` | must be `> 0` |
| `nu` | `float` | `1.0` | fixed to `1.0` internally (`eta = y`) |

### 5.6) TTE mechanism selection

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `tte_model` | `str` | `"auto"` | `auto | cox | aft` |
| `p_cox` | `float` | `0.5` | used only in `auto`, must be in `[0,1]` |

### 5.7) Censoring controls

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `censoring_mode` | `str` | `"auto"` | `auto | log_location | administrative` |
| `p_administrative_censoring` | `float` | `0.5` | used in auto mode, must be in `[0,1]` |
| `censoring_shape_concentration` | `float` | `2.0` | concentration for beta-range sampling, must be `> 0` |
| `censoring_log_location_family` | `str` | `"auto"` | `auto | normal | logistic | student_t` |
| `censoring_log_location_shift_min` | `float` | `-0.8` | lower bound for shift `b` |
| `censoring_log_location_shift_max` | `float` | `0.8` | upper bound for shift `b`, must be `>= min` |
| `censoring_log_location_student_df_min` | `float` | `4.0` | lower df bound if family is student_t, must be `> 2` |
| `censoring_log_location_student_df_max` | `float` | `20.0` | upper df bound, must be `>= min` |
| `censoring_admin_target_rate_min` | `float` | `0.2` | target censoring lower bound in `[0,1]` |
| `censoring_admin_target_rate_max` | `float` | `0.6` | target censoring upper bound in `[0,1]`, must be `>= min` |
| `censoring_admin_jitter_mode` | `str` | `"lognormal"` | `lognormal | uniform` |
| `censoring_admin_lognormal_sigma` | `float` | `0.15` | used for lognormal jitter, must be `>= 0` |
| `censoring_admin_uniform_radius` | `float` | `1.2` | used for uniform jitter in `[1/r, r]`, must be `>= 1` |
| `censoring_apply_guardrails` | `bool` | `True` | enable clamp to robust range |
| `censoring_clamp_min_multiplier` | `float` | `0.5` | lower guardrail multiplier, must be `> 0` |
| `censoring_clamp_max_multiplier` | `float` | `2.0` | upper guardrail multiplier, must be `> 0` |
| `censoring_time_min` | `float` | `1e-8` | absolute floor, must be `> 0` |
| `censoring_time_max` | `float` | `1e8` | absolute cap, must be `>= min` |

### 5.8) Cox controls

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `cox_tier` | `str` | `"auto"` | `auto | tier1 | tier2 | tier3 | tier4` |
| `cox_tier_probabilities` | `Tuple[float,float,float,float]` | `(0.55, 0.25, 0.15, 0.05)` | non-negative, length 4, sum `> 0` |
| `cox_weibull_theta_max` | `float` | `0.8` | Weibull `theta` max abs bound, must be `>= 0` |
| `cox_weibull_shape_concentration` | `float` | `2.0` | symmetric beta concentration, must be `> 0` |
| `cox_gompertz_hr_max` | `float` | `3.0` | hazard-ratio cap at reference time, must be `>= 1` |
| `cox_gompertz_reference_time` | `float` | `5.0` | positive reference time |
| `cox_gompertz_shape_concentration` | `float` | `2.0` | symmetric beta concentration, must be `> 0` |
| `cox_piecewise_min_intervals` | `int` | `3` | must be `>= 2` |
| `cox_piecewise_max_intervals` | `int` | `8` | must be `>= min_intervals` |
| `cox_piecewise_t_max` | `float` | `5.0` | positive time horizon |
| `cox_piecewise_breakpoint_alpha` | `float` | `2.0` | Dirichlet alpha for widths, must be `> 0` |
| `cox_piecewise_min_width_fraction` | `float` | `0.03` | min interval width fraction, must be `>= 0` |
| `cox_piecewise_b1_max` | `float` | `0.9` | coefficient max abs, must be `>= 0` |
| `cox_piecewise_b2_max` | `float` | `0.7` | coefficient max abs, must be `>= 0` |
| `cox_piecewise_b3_max` | `float` | `0.5` | coefficient max abs, must be `>= 0` |
| `cox_piecewise_shape_concentration` | `float` | `2.0` | symmetric beta concentration, must be `> 0` |
| `cox_mixture_min_components` | `int` | `2` | must be `>= 2` |
| `cox_mixture_max_components` | `int` | `3` | must be `>= min_components` |
| `cox_mixture_dirichlet_alpha` | `float` | `2.0` | Dirichlet alpha, must be `> 0` |
| `cox_mixture_component_families` | `Sequence[str]` | `(exponential, weibull, gompertz)` | non-empty subset of allowed Cox component families |

### 5.9) AFT controls

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `aft_tier` | `str` | `"auto"` | `auto | tier1 | tier2 | tier3 | tier4` |
| `aft_tier_probabilities` | `Tuple[float,float,float,float]` | `(0.55, 0.25, 0.15, 0.05)` | non-negative, length 4, sum `> 0` |
| `aft_shape_concentration` | `float` | `2.0` | beta-range concentration, must be `> 0` |
| `aft_sigma_min` | `float` | `0.6` | sigma lower bound, must be `> 0` |
| `aft_sigma_max` | `float` | `1.8` | sigma upper bound, must be `>= min` |
| `aft_student_df_min` | `float` | `3.0` | Student-t df lower bound, must be `> 2` |
| `aft_student_df_max` | `float` | `25.0` | Student-t df upper bound, must be `>= min` |
| `aft_gg_k_min` | `float` | `0.6` | generalized gamma `k` min, must be `> 0` |
| `aft_gg_k_max` | `float` | `2.0` | generalized gamma `k` max, must be `>= min` |
| `aft_gg_p_min` | `float` | `0.6` | generalized gamma `p` min, must be `> 0` |
| `aft_gg_p_max` | `float` | `2.0` | generalized gamma `p` max, must be `>= min` |
| `aft_gev_xi_max` | `float` | `0.3` | max abs xi bound, must be `>= 0` |
| `aft_skew_alpha_max` | `float` | `6.0` | max abs skew bound, must be `>= 0` |
| `aft_mixture_min_components` | `int` | `2` | must be `>= 2` |
| `aft_mixture_max_components` | `int` | `3` | must be `>= min_components` |
| `aft_mixture_dirichlet_alpha` | `float` | `2.0` | Dirichlet alpha, must be `> 0` |
| `aft_mixture_component_families` | `Sequence[str]` | `(normal, logistic, gumbel, student_t, skew_normal)` | non-empty subset of allowed AFT component families |

### 5.10) Runtime and preset

| Parameter | Type | Default | Allowed / Notes |
|---|---|---:|---|
| `seed` | `Optional[int]` | `None` | if set, seeds numpy and torch |
| `device` | `str` | `"cpu"` | torch device string |
| `difficulty` | `Optional[str]` | `None` | optional preset: `easy | medium | hard` |

## 6) Helper APIs for direct sampling

These are useful for unit tests and controlled generation.

### 6.1) Enumerations

- `available_tte_models()`
- `available_cox_baseline_tiers()`
- `available_cox_baseline_families()`
- `available_aft_tiers()`
- `available_aft_families()`
- `available_censoring_modes()`
- `available_log_location_censoring_families()`

### 6.2) ID mapping helpers

- `tte_model_to_id`, `tte_model_from_id`
- `cox_tier_to_id`, `cox_tier_from_id`
- `cox_baseline_family_to_id`, `cox_baseline_family_from_id`
- `aft_tier_to_id`, `aft_tier_from_id`
- `aft_family_to_id`, `aft_family_from_id`
- `censoring_mode_to_id`, `censoring_mode_from_id`
- `log_location_censoring_family_to_id`, `log_location_censoring_family_from_id`

### 6.3) Samplers

- `sample_tte_model(cfg, rng=None)`
- `sample_cox_baseline(cfg, rng=None)`
- `sample_aft_spec(cfg, rng=None)`
- `sample_event_times_cox(eta, cox_spec, rng=None)`
- `sample_event_times_aft(eta, aft_spec, cfg=None, rng=None)`
- `sample_censoring_mode(cfg, rng=None)`
- `sample_log_location_censoring_family(cfg, rng=None)`
- `sample_right_censoring(event_times, cfg, rng=None)`

## 7) Common recipes

### 7.1) Fully automatic mixed difficulty

```python
cfg = SimplifiedPriorConfig(
    generation_mode="auto",
    tte_model="auto",
    censoring_mode="auto",
    seed=123,
)
out = generate_simplified_prior_data(cfg, num_datasets=32)
```

### 7.2) Fixed Cox tier-3, moderate censoring control

```python
cfg = SimplifiedPriorConfig(
    tte_model="cox",
    cox_tier="tier3",
    censoring_mode="administrative",
    censoring_admin_target_rate_min=0.25,
    censoring_admin_target_rate_max=0.35,
    seed=123,
)
out = generate_simplified_prior_data(cfg, num_datasets=16)
```

### 7.3) Fixed AFT tier-4 mixture with log-location censoring

```python
cfg = SimplifiedPriorConfig(
    tte_model="aft",
    aft_tier="tier4",
    censoring_mode="log_location",
    censoring_log_location_family="student_t",
    censoring_log_location_shift_min=-0.3,
    censoring_log_location_shift_max=0.1,
    seed=123,
)
out = generate_simplified_prior_data(cfg, num_datasets=16)
```

### 7.4) Force around 30% censoring (administrative)

```python
cfg = SimplifiedPriorConfig(
    seq_len=200,
    censoring_mode="administrative",
    censoring_admin_target_rate_min=0.30,
    censoring_admin_target_rate_max=0.30,
    seed=42,
)
out = generate_simplified_prior_data(cfg, num_datasets=1)
```

Realized censoring in a finite sample can differ slightly from target.

### 7.5) Recommended bounded full distribution for pretraining

Use:
- `full_distribution_base_overrides()`: deterministic bounded defaults
- `sample_full_distribution_config(...)`: samples stage-dependent factors and returns a full config

```python
from simplified_prior import sample_full_distribution_config, generate_simplified_prior_data

cfg = sample_full_distribution_config(
    overrides={
        "seq_len": 512,
        "num_features": 20,
        "num_causes": 20,
        "seed": 123,
    }
)
out = generate_simplified_prior_data(cfg, num_datasets=16)
```

This profile is intentionally broad-but-bounded:
- includes all 3 generation modes (`head`, `causal`, `roots`) when feasible
- includes both TTE mechanisms (`cox`, `aft`)
- includes censoring diversity (`administrative`, `log_location`)
- tightens extreme tails compared with permissive defaults

Stage-dependent factor sampling in `sample_full_distribution_config`:
- generation mode:
  - values: `(\"head\", \"causal\", \"roots\")`
  - probs: `(0.45, 0.35, 0.20)`
  - if `num_causes != num_features`, `roots` is removed and probs are renormalized
- num layers:
  - values: `(3, 4, 5)`
  - probs: `(0.40, 0.40, 0.20)`
- hidden dim:
  - values: `(24, 32, 48)`
  - probs: `(0.35, 0.45, 0.20)`

Key bounded settings in `full_distribution_base_overrides()`:
- latent signal bounds:
  - `standardize_y=True`, `y_clip_value=8.0`
- Cox:
  - `cox_tier_probabilities=(0.45, 0.30, 0.20, 0.05)`
  - `cox_weibull_theta_max=0.6`
  - `cox_gompertz_hr_max=2.0`
  - piecewise tightened:
    - `min/max_intervals=3/6`
    - `t_max=4.0`
    - `b1/b2/b3 max = 0.7/0.5/0.3`
- AFT:
  - `aft_tier_probabilities=(0.45, 0.30, 0.20, 0.05)`
  - `sigma in [0.7, 1.4]`
  - `student_df in [5, 18]`
  - `gg_k, gg_p in [0.8, 1.5]`
  - `gev_xi_max=0.2`, `skew_alpha_max=4.0`
- Censoring:
  - `p_administrative_censoring=0.7`
  - admin target censoring rate in `[0.15, 0.45]`
  - log-location shift in `[-0.5, 0.5]`
  - guardrails enabled with tighter multipliers:
    - `censoring_clamp_min_multiplier=0.7`
    - `censoring_clamp_max_multiplier=1.6`
  - absolute time bounds:
    - `censoring_time_min=1e-6`
    - `censoring_time_max=1e6`

## 8) Validation checklist

If you build downstream pipelines, verify these invariants:

- `eta == y`
- `observed_T == min(T, C)`
- `delta` is binary (`0/1`)
- `event_indicators == (delta > 0.5)`
- `event_rate == mean(delta)` per dataset
- `censoring_rate == 1 - event_rate`
- all time fields (`T`, `C`, `observed_T`) are strictly positive

## 9) Notes on preset and curriculum

`difficulty` preset currently only sets stage-dependent architecture factors:
- `generation_mode`
- `num_layers`
- `hidden_dim`

All other fields remain explicit, stage-invariant controls unless you change them yourself.

## 10) Source of truth

This README mirrors the current code in:
- `simplified_prior/generator.py`
- `simplified_prior/__init__.py`

If code changes, update this manual accordingly.
