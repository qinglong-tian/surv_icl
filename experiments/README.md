# Factorial PMSE Experiment

Run a script-based factorial experiment (no notebooks) to quantify how prior factors affect dataset difficulty.

Difficulty is defined as PMSE (test MSE) averaged across common regression benchmarks.

## Command

```bash
python experiments/factorial_pmse_experiment.py
```

## Useful options

```bash
python experiments/factorial_pmse_experiment.py \
  --max-conditions 192 \
  --num-repeats 2 \
  --num-datasets-per-condition 2 \
  --output-dir experiments/results/factorial_pmse
```

- `--max-conditions <= 0` runs the full valid factorial grid.
- Positive `--max-conditions` runs a sampled fractional factorial subset for practicality.

## Outputs

CSV files are written to `experiments/results/factorial_pmse` (or your `--output-dir`):

- `raw.csv`: per-run PMSE by condition/model/repeat/dataset
- `condition_model.csv`: PMSE summary by condition and model
- `condition_difficulty.csv`: aggregated difficulty per condition
- `factor_importance.csv`: overall factor effect ranking
- `factor_importance_by_model.csv`: per-model factor effect ranking
- `metadata.csv`: run metadata
