# Multi-Turn Hybrid-Reasoning Guide

`sftmill` has first-class native support for distilling **multi-turn hybrid-reasoning** training data. This guide explains how hybrid reasoning works, how `sftmill` orchestrates multi-turn user-assistant dialogues, how internal chain-of-thought (`reasoning_content`) is captured and preserved across turns, and how to fine-tune open-weights models on the resulting datasets.

---

## 1. What is Hybrid Reasoning?

Modern reasoning models (such as DeepSeek-R1, OpenAI o1/o3-mini, and Qwen-2.5/3.5-Max) split their output into two distinct channels:

1. **Reasoning Scratchpad (`reasoning_content`)**: An internal, unbounded chain-of-thought where the model explores hypotheses, backtracks, verifies calculations, and plans its approach.
2. **User-Facing Response (`content`)**: The polished, helpful conversational reply delivered to the user.

In a **multi-turn** setting, human-AI interactions unfold sequentially:
```text
Turn 1: User asks a complex or underspecified question.
  └─ Assistant thinks (reasoning_content_1) -> responds with clarifying question or preliminary plan (content_1).

Turn 2: User provides clarification or additional constraints.
  └─ Assistant receives full prior context -> thinks through the updated requirements (reasoning_content_2) -> delivers final answer (content_2).
```

When training smaller models (e.g., 3B, 7B, 14B, 32B), distilling both the **internal chain-of-thought** and the **conversational prose** across multiple turns imparts strong conversational problem-solving skills without requiring multi-turn RL rollouts.

---

## 2. How `sftmill` Implements Multi-Turn Traces

`sftmill` splits dataset production into two clean stages:

```
┌─────────────────────────────────┐
│     Curriculum YAML Spec        │
│ (user_turns: 2 | 3, match: open)│
└────────────────┬────────────────┘
                 │
                 ▼  Stage 1: `sftmill tasks`
┌─────────────────────────────────────────────────────────┐
│                    Task Synthesis                       │
│  Synthesis Teacher creates realistic user turn sequences │
│        e.g., turns: ["Initial prompt", "Follow-up"]     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼  Stage 2: `sftmill generate`
┌─────────────────────────────────────────────────────────┐
│               Multi-Turn Rollout Loop                   │
│                                                         │
│   Turn 1:                                               │
│     append user_turn[0]                                 │
│     stream teacher -> capture reasoning_content + content│
│     append assistant response                           │
│                                                         │
│   Turn 2:                                               │
│     append user_turn[1]                                 │
│     stream teacher with prior context                   │
│     capture reasoning_content + content                 │
│     append assistant response                           │
│                                                         │
│   Acceptance Filter:                                    │
│     Verify non-stub prose (prose_ok, >= 2 sentences)    │
│     Save to sharded JSONL dataset                       │
└─────────────────────────────────────────────────────────┘
```

### Key Modules:
- **`sftmill.teachers.openai_compat`**: Streams Server-Sent Events (SSE) and decodes `delta.reasoning_content` (or non-streaming `choice.message.reasoning_content`), alongside standard `content`.
- **`sftmill.generate.traces._generate_open_trace`**: Iterates through each user turn, feeding previous assistant turns back to the teacher, capturing teacher reasoning for every turn.
- **`sftmill.generate.filters.prose_ok`**: Rejects lazy one-word responses, apologies, or empty stubs. Ensures assistant turns meet quality prose thresholds.

---

## 3. Curriculum Configuration for Multi-Turn Traces

To synthesize multi-turn hybrid-reasoning data, define a category with:
- `kind: trace`
- `match: open`
- `user_turns: 2` or `user_turns: 3`

### Example: Multi-Turn Architecture Consultation (2 Turns)

```yaml
name: architecture_curriculum
categories:
  - id: system_design_refactor
    count: 50
    kind: trace
    match: open
    user_turns: 2
    template: |
      JSON key: turns, a list of exactly 2 user messages and no assistant messages.
      The first turn asks how to refactor a monolithic component into microservices or async workers.
      The second turn introduces a strict constraint (e.g. latency under 50ms, zero-downtime database migration, or budget limit).
      No tools, no files, no answer key.
    example:
      turns:
        - "Our Django backend processes image uploads synchronously, causing HTTP request timeouts. How should we redesign this pipeline?"
        - "We are hosted on AWS and need to keep p99 latency under 200ms while processing 500 images/minute at minimum cost."
```

### Example: Multi-Turn Interactive Debugging (3 Turns)

```yaml
  - id: interactive_debugging
    count: 50
    kind: trace
    match: open
    user_turns: 3
    template: |
      JSON key: turns, a list of exactly 3 user messages and no assistant messages.
      The first turn reports an intermittent bug with a short stack trace.
      The second turn answers the assistant's diagnostic question by providing log outputs or environment details.
      The third turn asks for the final minimal patch and verification step.
      No tools, no files, no answer key.
    example:
      turns:
        - "Our FastAPI service randomly crashes with `RuntimeError: Task attached to a different loop`. What could cause this?"
        - "We are using Celery with asyncio tasks and a shared global SQLAlchemy async engine."
        - "That explains it! Show me how to safely manage the scoped async session lifecycle in Celery worker processes."
```

---

## 4. End-to-End Workflow

### Step 1: Synthesize Multi-Turn Task Blueprints

Run `sftmill tasks` with your curriculum YAML. You can use any high-capacity teacher (such as DeepSeek-V3/R1, GPT-4o, Claude 3.5 Sonnet via proxy, or Qwen-2.5-72B):

```bash
sftmill tasks \
  --curriculum configs/curriculum/code_instruct.yaml \
  --out data/tasks_instruct.jsonl \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key $OPENAI_API_KEY \
  --temperature 0.7 \
  --jobs 4
```

This writes rows to `data/tasks_instruct.jsonl`:
```json
{
  "id": "clarify-000001",
  "kind": "trace",
  "match": "open",
  "turns": [
    "Add a retry policy to our HTTP client.",
    "Exponential backoff up to 5 attempts, with jitter, targeting 5xx and 429 status codes only.",
    "Looks good. Now provide the complete Python implementation using httpx."
  ]
}
```

### Step 2: Mill Multi-Turn Hybrid-Reasoning Traces

Run `sftmill generate` pointing to a reasoning-capable teacher (such as DeepSeek-R1, o3-mini, Qwen-2.5-32B-Instruct with thinking enabled, etc.):

```bash
sftmill generate \
  --tasks data/tasks_instruct.jsonl \
  --out data/sft_dataset \
  --base-url http://localhost:8000/v1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --api-key none \
  --temperature 0.6 \
  --max-tokens 16384 \
  --kind trace \
  --jobs 6 \
  --shard-size 1000
```

---

## 5. Resulting Dataset Schema

Each generated row adheres to the canonical OpenAI Chat Completions messages schema, with `reasoning_content` embedded on every assistant turn:

```json
{
  "kind": "trace",
  "task_id": "clarify-000001",
  "messages": [
    {
      "role": "user",
      "content": "Add a retry policy to our HTTP client."
    },
    {
      "role": "assistant",
      "reasoning_content": "The user is requesting an HTTP retry policy, but hasn't specified the HTTP library (requests, httpx, aiohttp), maximum retry attempts, backoff strategy (constant, exponential, jitter), or which HTTP error status codes or network exceptions should trigger a retry. I should ask for these clarifications before writing code.",
      "content": "I would be happy to help implement a robust retry policy. Before writing the code, could you clarify:\n1. Which HTTP library are you using (`requests`, `httpx`, or `urllib3`)?\n2. What backoff strategy do you prefer (e.g., exponential backoff with full jitter)?\n3. What retry triggers do you want (e.g., 5xx server errors, 429 Rate Limited, connection timeouts)?"
    },
    {
      "role": "user",
      "content": "Exponential backoff up to 5 attempts, with jitter, targeting 5xx and 429 status codes only."
    },
    {
      "role": "assistant",
      "reasoning_content": "The user responded with concrete parameters: exponential backoff, max 5 attempts, jitter, targeting 5xx and 429. They didn't specify the library, so standard httpx or requests is appropriate. Let's ask if they want the complete Python implementation using httpx.",
      "content": "Got it: exponential backoff up to 5 attempts, full jitter, triggering on 5xx status codes and 429 Rate Limit responses. Would you like this implemented using `httpx` (with async or sync client support) or `requests`?"
    },
    {
      "role": "user",
      "content": "Looks good. Now provide the complete Python implementation using httpx."
    },
    {
      "role": "assistant",
      "reasoning_content": "Now we have all requirements: httpx client, sync and async support, exponential backoff (initial_delay * 2^attempt + uniform jitter), max 5 retries, checking response.status_code in (429, 500, 502, 503, 504) and httpx transport errors. Let's write clean, production-grade code with clear docstrings and error handling.",
      "content": "Here is a complete, production-ready implementation of an `httpx` retry transport using exponential backoff with full jitter:\n\n```python\nimport asyncio\nimport random\nimport time\nimport httpx\n\nRETRY_STATUS_CODES = {429, 500, 502, 503, 504}\n...\n```"
    }
  ]
}
```

---

## 6. Training with Hybrid-Reasoning Datasets

When fine-tuning student models on `sftmill` hybrid datasets, choose how to present `reasoning_content` to the tokenizer.

### Option A: Native Thinking Format (`<think> ... </think>`)

Popularized by DeepSeek-R1 and Qwen, you can merge `reasoning_content` directly into the assistant's content block:

```python
from sftmill.dataset import iter_jsonl_dataset
import json

def format_for_thinking_tokens(input_path: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as out:
        for row in iter_jsonl_dataset(input_path):
            formatted_messages = []
            for msg in row["messages"]:
                if msg["role"] == "assistant":
                    reasoning = msg.get("reasoning_content")
                    content = msg.get("content", "")
                    if reasoning:
                        full_content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                    else:
                        full_content = content
                    formatted_messages.append({"role": "assistant", "content": full_content})
                else:
                    formatted_messages.append(msg)
            out.write(json.dumps({"messages": formatted_messages}, ensure_ascii=False) + "\n")
```

### Option B: Training with HuggingFace TRL / Axolotl / Unsloth

Because `sftmill` outputs standard OpenAI chat-completion format JSONL, it works out-of-the-box with all leading fine-tuning frameworks:

- **Unsloth**: Use `dataset.map(formatting_prompts_func)` or the native chat template format.
- **Axolotl**: Set `type: chat_template` with `chat_template: chatml` or your model's custom Jinja template.
- **Llama-Factory**: Register as a multi-turn dataset with `messages` column.
- **HuggingFace TRL (`SFTTrainer`)**: Pass the JSONL directly with `dataset_text_field=None` and `formatting_func` utilizing `tokenizer.apply_chat_template`.

---

## 7. Quality Assurance & Filtering Rules

Multi-turn reasoning can suffer from common failure modes when using lower-quality teachers. `sftmill` implements automatic guards:

1. **Anti-Stub Filter (`prose_ok`)**: Discards assistant responses with fewer than two words or fewer than two grammatical sentences.
2. **Context Retention**: Prior conversation turns (both user messages and assistant responses with reasoning) are sent in the API message list so subsequent turns maintain coherent context.
3. **Max Token Protection**: If the teacher truncates mid-reasoning (`finish_reason == "length"`), the trace is safely rejected rather than saving corrupted training data.
