# Writing a curriculum

A curriculum YAML is the distribution `sftmill tasks` synthesizes from. This page is the authoring guide. The field list lives in [spec.md](spec.md).

## Shipped examples

Copy one of these and change the mix. Start with the smallest that matches the kind of data you want.

| File | What it is | Scale |
| --- | --- | --- |
| [`configs/curriculum/code_agent_test.yaml`](../configs/curriculum/code_agent_test.yaml) | Smoke copy of the agent mix. One harness (`alice`). | 16 categories, 2 questions each (~32 rows) |
| [`configs/curriculum/code_agent.yaml`](../configs/curriculum/code_agent.yaml) | Tool-use trajectories plus a few graded traces. Harness expansion. | 16 categories; trajectories multiply by harness count |
| [`configs/curriculum/code_instruct.yaml`](../configs/curriculum/code_instruct.yaml) | Coding-session chat. Open traces, no tools, files, or harnesses. | 22 categories × 80 = 1760 tasks |
| [`configs/curriculum/general_instruct.yaml`](../configs/curriculum/general_instruct.yaml) | Ordinary conversation plus identity. Most bins use the default student system; four ask the synthesizer for a custom `system`. | 80 categories / 4000 tasks |

`alice` on a trajectory category is a **tool-call envelope** (see [agent_trajectories.md](agent_trajectories.md)). The identity file [`configs/identity/alice.txt`](../configs/identity/alice.txt) is a **student system prompt**. They are not the same thing.

Open traces do not require `user_turns`. Use `user_turns: 2` or `3` only for multi-turn dialogues (`code_instruct`). Single-turn open (`general_instruct`) is valid.

## A curriculum is a distribution of categories

`sftmill tasks` synthesizes **one category at a time**. Grain is the category list, not the total `count`.

- Many thin bins keep intents distinct. `general_instruct` is the scale model: about 80 categories, most in the 40–80 row range, a few denser.
- A handful of fat bins collapse. The synthesizer copies the few-shot `example`, modes smear together, and identity / custom-system / “no tools” never show up as their own intents.

Total tasks = sum of `count` values. Trajectory categories with `harnesses` then expand: one question becomes one task per harness, sharing `group_id`.

Each category needs a narrow `id`, a `count`, a `template` that states the JSON keys and the intent, and an `example` in that shape. Tell the synthesizer not to reuse the example topic (“Use a new domain and register”).

## How to write one (LLM workflow)

Point a model at the example YAMLs, describe the mix you want, and make it write the file. Typical inputs:

- which examples to imitate (schema and grain)
- `kind` / `match` (trace vs trajectory, exact / contains / open)
- tools or not; identity (`student_system: default`, `custom`, or `off`)
- rough total rows
- what should be dense vs thin

**The usual failure is too few categories.** Tell the authoring model to use many narrow bins (dozens, not 4–8). For a wide instruct mix, aim near `general_instruct` (about 40–80 categories), not `code_agent_test`.

Copy-paste prompt:

```text
Look at these curriculum YAML files as schema and grain examples:
- configs/curriculum/code_agent_test.yaml  (tiny smoke)
- configs/curriculum/code_agent.yaml       (tool-use + harnesses)
- configs/curriculum/code_instruct.yaml    (coding-session chat)
- configs/curriculum/general_instruct.yaml (wide open instruct, 80 categories)

Then write a new curriculum YAML for this mix:

<describe kinds, match modes, tools or not, identity, total rows, what is dense vs thin>

Rules:
- Follow the field spec in docs/spec.md. Every category needs id, count, kind, match, template, and example.
- Use many categories (dozens, not a handful). Each category is one narrow intent.
- Prefer many thin bins over a few large ones so synthesis does not collapse to the example topic.
- For a wide instruct mix, 40–80 categories is the right grain; general_instruct is the scale model.
- Template text must say “Use a new domain and register. Do not reuse the example topic.”
- Total tasks = sum of counts (times harnesses if set). Do not invent fields the spec does not list.
```

## After the YAML exists

```bash
sftmill tasks \
  --curriculum configs/curriculum/your_mix.yaml \
  --out tasks.jsonl \
  --model <teacher> \
  --base-url http://127.0.0.1:8080/v1 --jobs 4 \
  --base-url http://127.0.0.1:8081/v1 --jobs 4

sftmill generate \
  --tasks tasks.jsonl \
  --out data/sft_output \
  --kind both \
  --model <teacher> \
  --base-url http://127.0.0.1:8080/v1 --jobs 4 \
  --base-url http://127.0.0.1:8081/v1 --jobs 4
```

`--base-url` and `--jobs` can be repeated (same `--model`). One `--jobs` applies to every URL; repeating `--jobs` once per URL sets that server. Default is 3 jobs per URL. See the CLI tables in the README.

Open traces inject [`configs/identity/alice.txt`](../configs/identity/alice.txt) at generate unless the category set `student_system: custom` or `off`. For `default`, generate re-reads the live identity file rather than a stale `system` baked into the task.
