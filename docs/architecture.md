# Architecture & System Design

This document details the internal design and data flow of `sftmill`.

---

## 1. High-Level Architecture

`sftmill` is designed as a decoupled two-stage data distillation pipeline:

```
                  ┌──────────────────────────────┐
                  │   Curriculum Blueprint       │
                  │   (YAML Specification)       │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     sftmill tasks     │
                     │  (Task Synthesizer)   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │          tasks.jsonl         │
                  │  - Graded problem statements │
                  │  - Ground-truth answers      │
                  │  - Seed workspaces & checks  │
                  │  - Multi-turn turn sequences │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │    sftmill generate   │
                     │   (Rollout Engine)    │
                     └───────────┬───────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │   Traces (CoT)    │           │   Trajectories    │
       │ Multi-turn prose  │           │ Sandboxed Tools   │
       │ & reasoning steps │           │ Automated checks  │
       └─────────┬─────────┘           └─────────┬─────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Grading & Filtering   │
                    │  - exact / contains     │
                    │  - test pass assertions │
                    │  - non-stub prose       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Production SFT Dataset    │
                  │  - Auto-sharded JSONL files  │
                  │  - Standard OpenAI messages  │
                  │  - Group-split ready         │
                  └──────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Curriculum Loader (`sftmill.generate.synthesize_tasks.load_curriculum`)
- Parses and strictly validates YAML curricula.
- Enforces presence of required keys (`id`, `count`, `kind`, `match`, `template`).
- Validates `user_turns` (2 or 3 when set), `student_system` (`default` / `custom` / `off`), match modes, and harnesses.

### 2.2 Task Synthesizer (`sftmill.generate.synthesize_tasks`)
- Uses thread pool execution (`ThreadPoolExecutor`) across categories.
- Each in-flight category **borrows** one slot from `TeacherPool` (one client object per job, possibly on different `--base-url`s).
- Batch synthesizes tasks using structured JSON output prompts and few-shot examples.
- Automatically repairs malformed JSON using iterative prompt repair (`MAX_REPAIR_TURNS = 2`).
- Expands multi-harness variants into linked task sibling rows sharing `group_id`.
- Optional category `student_system`: `default` prepends [`configs/identity/alice.txt`](../configs/identity/alice.txt); `custom` requires a synthesized `system` string; `off` skips it.

### 2.3 Teacher Client (`sftmill.teachers.openai_compat`)
- Native Python standard library implementation (`urllib.request`) with zero heavy SDK dependencies.
- `OpenAICompatibleTeacher` handles both streaming Server-Sent Events (SSE) and buffered completions.
- `TeacherPool` is a fixed set of slots. `borrow()` blocks until a slot is free, then returns it.
- Repeat `--base-url` to create slots on more than one server; `--jobs` is the slot count per URL (default 3).
- Live token-level streaming display to stdout (discarded when total jobs > 1 so workers do not mix output).
- Extracts both `content` and `reasoning_content` from streaming deltas and non-streaming responses.
- Assembles fragmented tool calls across streaming chunks.

Open `generate` traces save the student system on the SFT row. The teacher call uses that text plus an overlay (`TEACHER_OPEN_INSTRUCT`: no repo/files unless the user provided them; no tool XML). `student_system: default` re-reads the live identity file at generate. Exact/contains traces still send `task["messages"]` as written.

### 2.4 Hermetic Workspace Sandbox (`sftmill.tools.workspace.Workspace`)
- Creates temporary directories for each trajectory rollout.
- Seeds directory structures and files from task definitions.
- Intercepts all file operations (`list_dir`, `read_file`, `write_file`, `edit_file`, `search`, `bash`).
- Enforces boundary restrictions: canonicalizes paths and rejects any attempt to traverse outside the workspace sandbox.

### 2.5 Python Sandbox (`sftmill.tools.python_sandbox.run_python`)
- Executes Python scripts in isolated subprocesses.
- Runs with isolated environment flags (`-I`), scrubbed `PATH`, disabled bytecode caching, and stripped GPU environment variables (`CUDA_VISIBLE_DEVICES=""`).
- Hard 5.0-second timeout enforcement.

### 2.6 Graded Acceptance Filters (`sftmill.generate.filters`)
- **`exact`**: Compares the assistant's final response with the expected answer.
- **`contains`**: Checks whether the target answer exists in the assistant response or reasoning trace.
- **`open`**: Validates conversational multi-turn responses, verifying that assistant turns contain substantive prose rather than stub replies.
- Trajectory validation: Runs `verify` command in the workspace and checks `require_observation` tokens.

### 2.7 Sharded Dataset Writer (`sftmill.dataset.JsonlDatasetWriter`)
- Thread-safe append-only file writing with internal mutex locking.
- When `--out` is a directory, automatically segments datasets into numbered shards (`part-000000.jsonl`, `part-000001.jsonl`, etc.) upon reaching `--shard-size` (default 1,000 rows).
- Guarantees immediate flush per row to prevent data loss on interrupted runs.

### 2.8 Real-Time Progress Reporter (`sftmill.progress.ProgressReporter`)
- Background thread printing periodic status metrics every `--progress-interval` seconds (default 30s):
  ```text
  [generate] total=100 processed=42 kept=38 failed=4 written=38 (last=fix_bug-0012)
  ```
- Gracefully handles SIGINT/SIGTERM without corrupting open files.

---

## 3. Data Integrity & Leak Prevention

A persistent risk in SFT data generation is evaluation leakage when tasks are mutated across multiple harnesses or multiple iterations.

`sftmill` addresses this through **Group Hashing**:
1. When generating multiple harness representations of a task (e.g., `openai`, `claude_code`, `alice`), all sibling rows inherit `group_id`.
2. The `sftmill.dataset.split_by_group` utility hashes the `group_id` via SHA-256:
   $$\text{bucket} = \frac{\text{int}(\text{SHA256}(\text{seed} : \text{group\_id})[:8], 16)}{2^{32} - 1}$$
3. All sibling rows are assigned to the exact same partition (train or validation), preventing memorization leakage.
