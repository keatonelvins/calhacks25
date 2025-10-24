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
uv run train.py --config configs/wordle.toml --detach
```

Artifacts will be stored in the `vf-wordle` volume!

# Refs
- https://github.com/PrimeIntellect-ai/verifiers/tree/main/verifiers/rl
- https://modal.com/docs/guide/volumes