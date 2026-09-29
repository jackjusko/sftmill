# Curriculum spec (`--curriculum`)

A curriculum YAML drives `sftmill tasks`. It tells a synthesis teacher what kinds of rows to invent; `sftmill generate` then runs a separate teacher over the resulting `tasks.jsonl`.

Worked examples (start with the small one):

- [`configs/curriculum/code_agent_test.yaml`](../configs/curriculum/code_agent_test.yaml)
- [`configs/curriculum/code_agent.yaml`](../configs/curriculum/code_agent.yaml)
- [`configs/curriculum/code_instruct.yaml`](../configs/curriculum/code_instruct.yaml)

This document lists only fields that `load_curriculum` accepts. There is no Distill-style run, model, or hardware YAML in sftmill.

## File shape

```yaml
name: my_spec          # optional label
categories:
  - id: example
    count: 10          # positive integer; rows to synthesize in this category
    kind: trace        # trace | trajectory
    match: exact       # exact | contains | open
    template: |
      Instructions to the synthesis teacher…
```

Every category **must** include `id`, `count`, `kind`, `match`, and `template`.

## `kind`

| Value | Meaning |
| --- | --- |
| `trace` | No tools. User message(s), then teacher completions (reasoning + answer). |
| `trajectory` | Tools enabled. Teacher may emit `tool_calls`; sftmill runs the sandbox/workspace and appends observations until a final answer. |

## `match`

| Value | Meaning |
| --- | --- |
| `exact` | Gold `answer` must match the graded final assistant value. |
| `contains` | Gold answer is a needle in the reply (often one token, e.g. an exception name). |
| `open` | No gold `answer`. Replies must be real prose (not a one-token stub). Use with multi-turn dialogue when needed. |

For `match: open`, optional `user_turns` must be **2** or **3**. Synthesis returns `turns` (user strings), not `answer`.

## Optional category keys

| Key | Purpose |
| --- | --- |
| `example` | Few-shot item shown during synthesis (same JSON shape as a synthesized row). |
| `max_steps` | Positive integer copied onto each task; caps tool rounds in `generate`. |
| `harnesses` | List of `alice`, `openai`, `claude_code`, `novel`. One question becomes several tasks sharing `group_id`. **`alice` is a tool-call envelope id, not a model checkpoint.** |
| `toolset` | `workspace` (canonical file/bash/python tools) or `custom` (teacher must emit OpenAI function schemas in `tools`). Omit for traces. |
| `verify` | Shell command run in the seeded workspace after a trajectory (e.g. `python3 check.py`). |
| `verify_stdout` | Set to `exact` when stdout must equal the gold `answer`. |
| `verify_on` | Only legal value: `seed` — grade against original files (lookup tasks); no `expect_files`. |
| `require_observation` | Substring that must appear in a tool observation (example specs often use `PASSED`). |
| `user_turns` | `2` or `3` for open multi-turn categories. |

## Synthesis item shape (what the teacher returns)

Depends on the category:

- **All graded traces/trajectories:** `question`, `answer` (except `match: open`).
- **Open multi-turn:** `turns` — list of user strings; no gold `answer`.
- **`toolset: workspace`:** `files` (path → text) and usually `expect_files` (gold tree after edits).
- **`verify_on: seed`:** `verify` — command whose stdout is the answer; no `expect_files`.
- **`toolset: custom`:** `tools` — OpenAI-style function list (one useful tool + decoy is typical).
- **Category id `long_feature` only:** also `correction` — one sentence injected mid-trajectory; validated specially in code. Do not assume other ids accept `correction`.

## Minimal examples

Trace with a gold answer:

```yaml
categories:
  - id: needle
    count: 5
    kind: trace
    match: contains
    template: |
      JSON keys: question, answer.
      answer is one token: an exception class.
      question is a short snippet that triggers it.
    example:
      question: |
        def f(d): return d["k"]
        f({})
      answer: KeyError
```

Open instruct (no tools):

```yaml
categories:
  - id: chat
    count: 20
    kind: trace
    match: open
    user_turns: 2
    template: |
      JSON key: turns — exactly 2 user messages, no assistant text, no answer.
    example:
      turns:
        - What does this function do?
        - Shorter, please.
```

Workspace trajectory:

```yaml
categories:
  - id: fix_bug
    count: 10
    kind: trajectory
    match: exact
    max_steps: 16
    toolset: workspace
    require_observation: PASSED
    verify: python3 check.py
    harnesses: [openai, claude_code]
    template: |
      JSON keys: question, answer, files, expect_files.
      answer is the single token ok.
```

## Output

`sftmill tasks` writes `tasks.jsonl`. `sftmill generate` writes trace/trajectory rows: OpenAI-style `messages`, optional `tools`, optional `harness`, optional `group_id`. See [`configs/tasks/examples.jsonl`](../configs/tasks/examples.jsonl) for tiny task rows.
