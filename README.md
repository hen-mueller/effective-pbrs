# Designing Effective Potential-based Reward Shaping

## Running experiments

Use `run_experiment.py` to launch an experiment:

```bash
python run_experiment.py <env> <log_dir> --activate_shaping [other options]
```

Relevant shaping reset options:

- `--reset_to_bias`: set potential of terminal states to `--reset_value` instead of zero.

