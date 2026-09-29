# Agent Trajectories & Sandboxed Tool Use Guide

`sftmill` provides a verifiable execution environment for distilling autonomous software engineering and tool-use agent trajectories (`kind: trajectory`).

Instead of collecting unverified model rollouts that hallucinate file edits and test results, `sftmill` executes tool calls in **hermetic sandboxes**, runs verification commands, and only commits trajectories that pass automated assertions.

---

## 1. How Agent Trajectories Work

When `sftmill generate` processes a trajectory task:

```
┌────────────────────────────────────────────────────────┐
│                      Task Input                        │
│  - Seed files (e.g. app.py, lib.py, check.py)          │
│  - User question ("Fix the bug. Run check.py.")        │
│  - Gold target: expect_files or verify command         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │  Isolated Workspace     │
               │  tempfile sandbox       │
               │  seeded with files      │
               └────────────┬────────────┘
                            │
 ┌──────────────────────────┴──────────────────────────┐
 │                                                     ▼
 │                   Step Loop (up to max_steps)
 │
 │  1. Teacher completes with tool_calls (or reasoning)
 │  2. Workspace executes tool:
 │       - list_dir, read_file, write_file, edit_file,
 │         search, bash, or python
 │  3. Observation is formatted & appended as role: tool
 │  4. Loop repeats until assistant emits final text
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │    Verification Phase     │
              │  - verify command passes  │
              │  - expect_files match     │
              │  - require_observation    │
              │  - match exact/contains   │
              └─────────────┬─────────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
            [PASS]                    [FAIL]
               │                         │
               ▼                         ▼
        Write to JSONL            Discard rollout
        (with chosen harness)     (progress.bump('failed'))
```

---

## 2. Toolsets: Canonical Workspace Tools

When a task specifies `toolset: workspace`, `sftmill` equips the agent with 7 canonical software engineering tools:

| Tool | Parameters | Description |
|---|---|---|
| `python` | `code: string` | Runs Python in an isolated subprocess with a 5-second timeout. |
| `list_dir` | `path: string` | Lists directory contents. Guards against path traversal (`..`). |
| `read_file` | `path: string, offset?: int, limit?: int` | Reads text with optional line paging. |
| `write_file` | `path: string, content: string` | Creates or overwrites a file in the workspace. |
| `edit_file` | `path: string, old: string, new: string` | Replaces exactly one occurrence of `old` with `new`. Rejects ambiguous edits. |
| `search` | `pattern: string, path?: string` | Performs literal string search across workspace files. Returns `path:line:text`. |
| `bash` | `command: string` | Executes a shell command inside the workspace directory with scrubbed environment variables. |

### Sandboxing & Security Safeguards

- **No Path Escapes**: Every path is checked via `Path.resolve()`. Any reference attempting to escape the temporary workspace (`..` or root traversal) returns an explicit error string rather than executing.
- **Clean Environment**: Subprocesses (`bash` and `python`) execute with scrubbed environment variables:
  - `CUDA_VISIBLE_DEVICES=""` (prevents accidental GPU memory allocation during datagen)
  - `PYTHONNOUSERSITE="1"` (prevents loading untrusted packages from user home)
  - `PYTHONDONTWRITEBYTECODE="1"`
  - `PATH="/usr/bin:/bin"`
- **Timeout Caps**: Python code snippets time out after 5 seconds.
- **Output Capping**: Tool outputs exceeding `max_output` (default 2,000 characters) are deterministically truncated with `[truncated]`.
- **Loop Detection**: If an agent emits identical tool calls 3 times in a row, the trajectory is immediately aborted to prevent infinite loops.

---

## 3. Automated Verification Modes

`sftmill` supports multiple layers of ground-truth verification:

### Mode 1: Unit Test Verification (`verify` + `require_observation`)

The task seeds a workspace containing a test script (e.g., `check.py`) and buggy code. The test script asserts on correct behavior and prints `PASSED`.

```yaml
  - id: fix_caching_bug
    count: 20
    kind: trajectory
    match: exact
    max_steps: 16
    toolset: workspace
    require_observation: PASSED
    verify: python3 check.py
    template: |
      JSON keys: question, answer, files, expect_files.
      files has app.py and check.py.
      check.py fails initially, and prints PASSED only after app.py is fixed.
      expect_files holds the fixed app.py.
      answer is the single token ok.
```

When generating:
1. `require_observation: PASSED`: The teacher MUST have executed a tool (such as `bash: python3 check.py`) that returned `PASSED` during the rollout.
2. `verify: python3 check.py`: After the agent terminates, `sftmill` independently runs `python3 check.py` in the workspace.
3. `expect_files`: `sftmill` verifies that the final workspace files exactly match the gold `expect_files` dictionary.

### Mode 2: Read-Only Investigation (`verify_on: seed`)

For tasks where the model investigates a codebase and computes an answer without modifying files (e.g., searching for a bug root cause):

```yaml
  - id: find_slow_query
    count: 15
    kind: trajectory
    match: exact
    toolset: workspace
    verify_on: seed
    verify: python3 -c "import repo; print(repo.slowest_query_id())"
    template: |
      JSON keys: question, answer, files.
      question asks which query ID causes the N+1 problem.
      verify prints the exact query ID.
```

### Mode 3: Custom Toolsets (`toolset: custom`)

For general agent tasks that don't need workspace file operations (e.g. interacting with an API or mock database):

```yaml
  - id: database_lookup
    count: 25
    kind: trajectory
    match: exact
    toolset: custom
    template: |
      JSON keys: question, answer, tools.
      tools provides OpenAI function definitions: one useful tool and one decoy.
```

---

## 4. Multi-Harness Translation

A critical challenge in training generalist agents is that different runtime platforms format tool calls differently.

Instead of retraining or regenerating data for each platform, `sftmill` decouples the trajectory rollout from its serialized envelope using **harness specifications**:

| Harness ID | Target Platform | Call Format | Observation Format |
|---|---|---|---|
| `openai` | OpenAI Assistants / Function Calling | `{"name":"write_file","arguments":{...}}` | `tool write_file: ok` |
| `claude_code` | Claude Code / Anthropic XML | `<tool_call><tool_name>Write</tool_name><arguments>...</arguments></tool_call>` | `<tool_result name="Write">ok</tool_result>` |
| `alice` | Jinja-delimited native envelopes | `<write_file>\n{...}\n</write_file>` | `OBSERVE write_file: ok` |
| `novel` | Out-of-distribution evaluation | `[[nova_write]]\n{...}` | `[[/nova_write]]\nok` |

### Multi-Harness Expansion

When you specify `harnesses: [openai, claude_code]` in a category YAML:
- One synthesized question automatically expands into two tasks:
  - `task-0001-openai`
  - `task-0001-claude_code`
- Both share `group_id: task-0001`.
- When generating, the agent's tool calls and tool observation headers are automatically rendered in the corresponding target format!

---

## 5. Leak-Free Train / Validation Splitting

When evaluating agent performance across harnesses, it is fatal if `task-0001-openai` ends up in the training set while `task-0001-claude_code` ends up in the validation set (the model would memorize the question).

`sftmill` provides built-in group-aware deterministic splitting:

```python
from sftmill.dataset import iter_jsonl_dataset, split_by_group

# Load all generated rows
rows = list(iter_jsonl_dataset("data/sft_dataset"))

# Split 90% train / 10% val, guaranteeing sibling harnesses never cross splits
train_rows, val_rows = split_by_group(rows, val_fraction=0.1, seed=42)

print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")
```
Each group's SHA-256 hash determines split placement, ensuring rigorous empirical hygiene.
