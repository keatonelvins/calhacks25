# Setup
- Make an account at https://modal.com, then run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run modal setup
uv run modal create wandb-secret WANDB_API_KEY=...
```

# Train:
```bash
uv run modal run train.py
uv run modal run --detach train.py  # or detach for long runs
```

# Refs
- https://github.com/PrimeIntellect-ai/verifiers/tree/main/verifiers/rl