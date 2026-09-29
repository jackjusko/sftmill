"""Keep teacher rows that match the task's answer check."""

from __future__ import annotations

import logging
import re

from sftmill.generate.interpret import extract_graded_answer

logger = logging.getLogger("sftmill.generate.filters")


def short_answer(answer: str) -> str:
    """Use the value after the last ``Result:`` when the task stored a whole script."""
    text = answer.strip()
    if "Result:" in text:
        return text.rsplit("Result:", 1)[-1].strip()
    return text


def _assistant_blob(example: dict) -> str:
    parts: list[str] = []
    for message in example["messages"]:
        if message.get("role") != "assistant":
            continue
        parts.append(message.get("content") or "")
        parts.append(message.get("reasoning_content") or "")
    return "\n".join(parts)


def final_answer(example: dict, task: dict | None = None) -> str:
    expected = None if task is None else task.get("answer")
    if expected is not None:
        expected = str(expected).strip()
    for message in reversed(example["messages"]):
        if message["role"] == "assistant":
            graded = extract_graded_answer(message, expected)
            if graded:
                return graded
    return ""


def _final_content(example: dict) -> str:
    for message in reversed(example["messages"]):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return (message.get("content") or "").strip()
    return ""


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def prose_ok(text: str) -> bool:
    """An open reply is more than one token and at least two sentences."""
    stripped = text.strip()
    if len(stripped.split()) < 2:
        return False
    return _sentence_count(stripped) >= 2


def _open_ok(example: dict) -> bool:
    assistants = [message for message in example["messages"] if message.get("role") == "assistant"]
    if not assistants:
        return False
    for message in assistants:
        if message.get("tool_calls"):
            return False
        if not prose_ok(message.get("content") or ""):
            return False
    return True


def accept(example: dict, task: dict) -> bool:
    mode = task.get("match", "exact")
    if mode == "open":
        ok = _open_ok(example)
        if not ok:
            logger.warning("answer mismatch task=%s open reply was not prose", task.get("id") or task.get("task_id"))
        return ok
    if task.get("answer") is None:
        return True
    expected = short_answer(str(task["answer"]).strip())
    if mode == "exact":
        got = _final_content(example)
        ok = got == expected or short_answer(got) == expected
    elif mode == "contains":
        got = final_answer(example, task).strip()
        ok = expected in got or expected in _assistant_blob(example)
    else:
        raise ValueError(f"unknown match mode {mode}")
    if not ok:
        preview = got if len(got) <= 120 else got[:117] + "..."
        logger.warning(
            "answer mismatch task=%s expected=%r got=%r",
            task.get("id") or task.get("task_id"),
            expected,
            preview,
        )
    return ok
