# Contributing

## Setup

```bash
cd /path/to/sftmill
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

sftmill does not use CUDA. The suite clears `CUDA_VISIBLE_DEVICES` unless `SFTMILL_GPU_TESTS=1`.

```bash
pytest
```

## Curriculum docs

See [`docs/spec.md`](docs/spec.md) for the `--curriculum` YAML reference.
