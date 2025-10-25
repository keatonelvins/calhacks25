# Setup
- Make an account at https://modal.com, then run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run modal setup
uv run modal create wandb-secret WANDB_API_KEY=...
```

# Train:
```bash
uv run train.py --help
uv run train.py --config configs/alphabet-sort.toml
uv run train.py --config configs/wordle.toml --detach
```

# Build:
```bash
uv run vf-init env-name
uv run vf-install env-name
uv run train.py --config configs/env-name.toml
```

Artifacts will be stored in the `vf-artifacts` volume (see `uv run modal volume --help`).

# Refs
- https://github.com/PrimeIntellect-ai/verifiers/tree/main/verifiers/rl
- https://modal.com/docs/guide/volumes