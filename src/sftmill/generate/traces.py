"""One teacher completion becomes one reasoning trace."""

from __future__ import annotations

import logging

from sftmill.generate.filters import accept
from sftmill.generate.interpret import extract_graded_answer, interpret_teacher
from sftmill.schema import validate_example

logger = logging.getLogger("sftmill.generate.traces")


def _user_turns(task: dict) -> list[str]:
    turns = task.get("turns")
    if isinstance(turns, list) and turns:
        return [str(turn) for turn in turns]
    return [
        str(message.get("content") or "")
        for message in task["messages"]
        if message.get("role") == "user"
    ]


def _open_assistant(raw) -> dict | None:
    if isinstance(raw, str):
        content, reasoning = raw, ""
    else:
        if raw.get("finish_reason") == "length":
            return None
        content = raw.get("content") or ""
        reasoning = raw.get("reasoning_content") or ""
    assistant = {"role": "assistant", "content": str(content).strip()}
    if str(reasoning).strip():
        assistant["reasoning_content"] = str(reasoning).strip()
    if not isinstance(raw, str) and raw.get("tool_calls"):
        assistant["tool_calls"] = raw["tool_calls"]
    return assistant


def _generate_open_trace(task: dict, teacher) -> dict | None:
    task_id = task.get("id", "?")
    messages: list[dict] = []
    for turn in _user_turns(task):
        messages.append({"role": "user", "content": turn})
        raw = teacher.complete(messages)
        assistant = _open_assistant(raw)
        if assistant is None:
            logger.warning("trace rejected task=%s teacher hit max_tokens", task_id)
            return None
        messages.append(assistant)
    example = {
        "kind": "trace",
        "task_id": task_id,
        "messages": messages,
    }
    if task.get("group_id"):
        example["group_id"] = task["group_id"]
    if not accept(example, task):
        logger.warning("trace rejected task=%s (see answer mismatch above)", task_id)
        return None
    logger.info("trace kept task=%s", task_id)
    return validate_example(example)


def generate_trace(task: dict, teacher) -> dict | None:
    if task.get("match") == "open":
        return _generate_open_trace(task, teacher)
    task_id = task.get("id", "?")
    logger.info("trace task=%s", task_id)
    raw = teacher.complete(task["messages"])
    if isinstance(raw, str):
        merged = {"content": raw}
    else:
        merged = dict(raw)
    if not isinstance(raw, str) and raw.get("finish_reason") == "length":
        logger.warning("trace rejected task=%s teacher hit max_tokens", task_id)
        return None
    assistant = interpret_teacher(raw)
    if task.get("match") == "contains" and task.get("answer") is not None:
        assistant["content"] = extract_graded_answer(merged, str(task["answer"]))
    example = {
        "kind": "trace",
        "task_id": task_id,
        "messages": list(task["messages"]) + [assistant],
    }
    if task.get("group_id"):
        example["group_id"] = task["group_id"]
    if task.get("answer") is not None:
        example["answer"] = task["answer"]
    if not accept(example, task):
        logger.warning(
            "trace rejected task=%s (see answer mismatch above)",
            task_id,
        )
        return None
    logger.info("trace kept task=%s", task_id)
    return validate_example(example)


def generate_traces(tasks: list[dict], teacher):
    kept = 0
    total = sum(1 for task in tasks if task.get("kind", "trace") == "trace")
    for task in tasks:
        if task.get("kind", "trace") != "trace":
            continue
        row = generate_trace(task, teacher)
        if row is not None:
            kept += 1
            yield row
    logger.info("traces kept %s/%s", kept, total)
