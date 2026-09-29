# Contributing to sftmill

We welcome contributions to `sftmill`! Whether you're adding support for new curriculum templates, optimizing teacher streaming, extending sandbox tools, or improving documentation, here is how to get started.

---

## 🛠️ Development Setup

`sftmill` requires Python 3.10 or higher. You can use standard `venv` or `uv`:

### Option A: Using `uv` (Recommended)

```bash
cd sftmill
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Option B: Using standard Python `venv`

```bash
cd sftmill
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 🧪 Running Tests

The test suite runs with `pytest` and verifies task synthesis, trajectory execution, workspace sandbox isolation, progress tracking, and dataset serialization.

`sftmill` does not require or use CUDA during synthesis or test execution. The suite automatically clears `CUDA_VISIBLE_DEVICES` to prevent unintended GPU memory allocation.

```bash
# Run the test suite with pytest
pytest

# Or via uv without manual venv activation
uv run --with pytest pytest
```

---

## 📁 Repository Structure

```text
src/sftmill/
├── cli.py                   # Main CLI entrypoint (tasks, generate)
├── dataset.py               # Sharded JSONL reader, writer, and group splitter
├── harness.py               # Envelope definitions (alice, openai, claude_code, novel)
├── log.py                   # Logging configuration and verbosity
├── progress.py              # Background progress thread and status reporting
├── schema.py                # Schema validators and tool definitions
├── generate/
│   ├── filters.py           # Graded answer matching and prose verification
│   ├── interpret.py         # Teacher parsing, reasoning extraction, call normalization
│   ├── synthesize_tasks.py  # Stage 1: Curriculum YAML to tasks.jsonl
│   ├── traces.py            # Stage 2A: Reasoning traces and multi-turn dialogues
│   └── trajectories.py      # Stage 2B: Sandboxed tool execution and verification
├── teachers/
│   └── openai_compat.py     # Streaming & buffered client for any OpenAI endpoint
└── tools/
    ├── python_sandbox.py    # Isolated, timeout-capped Python runner
    └── workspace.py         # Hermetic filesystem sandbox (file operations, bash, search)
```

---

## 📋 Pull Request Guidelines

1. **Tests Pass**: Ensure all existing tests pass (`pytest`) and add test coverage for new functionality under `tests/`.
2. **Deterministic Sandboxing**: When adding tools or sandbox modifications, verify paths cannot escape the temporary workspace root.
3. **Clean Dependencies**: Keep runtime dependencies minimal. `sftmill` core only depends on `pyyaml`.
4. **Documentation**: Update relevant docs under `docs/` and `README.md` if your change modifies CLI arguments or YAML curriculum schemas.

---

## 📖 Documentation Reference

- [Curriculum Specification](docs/spec.md)
- [Multi-Turn Hybrid-Reasoning Guide](docs/multi_turn_hybrid_reasoning.md)
- [Agent Trajectories & Sandboxed Tool Use](docs/agent_trajectories.md)
- [Architecture & System Design](docs/architecture.md)
