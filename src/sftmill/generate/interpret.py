"""Turn a teacher message into an assistant turn."""

from __future__ import annotations

import json
import re


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _strip_answer_wrappers(text: str) -> str:
    cleaned = text.strip().strip(".")
    lowered = cleaned.lower()
    for prefix in (
        "final answer:",
        "answer:",
        "the answer is",
        "therefore, the answer is",
        "therefore the answer is",
    ):
        if lowered.startswith(prefix):
            return cleaned[len(prefix):].strip().strip(":.").strip()
    return cleaned


def _tail_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


_NUMERIC_ANSWER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _strip_markdown(text: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", text)


def _numbers_in(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", _strip_markdown(text))


def _numeric_in_blob(text: str, expected: str) -> str | None:
    if not text or not _NUMERIC_ANSWER.fullmatch(expected):
        return None
    for match in re.finditer(r"\\boxed\{([^}]+)\}", text):
        inner = _strip_markdown(match.group(1)).strip()
        if inner == expected or expected in _numbers_in(inner):
            return expected
    if expected in _numbers_in(text):
        return expected
    return None


def align_answer(got: str, expected: str | None) -> str:
    """Map a verbose teacher line onto the task key when possible."""
    cleaned = _strip_answer_wrappers(_strip_markdown(got.strip()))
    if not expected:
        return cleaned
    expected = expected.strip()
    if cleaned == expected:
        return expected
    if _NUMERIC_ANSWER.fullmatch(expected):
        for blob in (cleaned, got):
            numbers = _numbers_in(blob)
            if expected in numbers:
                return expected
            if numbers and numbers[-1] == expected:
                return expected
    return cleaned


def extract_graded_answer(message: dict, expected: str | None = None) -> str:
    """Pull the string we compare to the task's ``answer`` key."""
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    if expected is not None:
        needle = str(expected).strip()
        if needle and (needle in content or needle in reasoning):
            return needle
    for blob in (content, reasoning, f"{reasoning}\n{content}".strip()):
        if not blob:
            continue
        parsed = _json_object(blob)
        if parsed and parsed.get("answer") is not None:
            return align_answer(str(parsed["answer"]), expected)
    if expected and _NUMERIC_ANSWER.fullmatch(expected.strip()):
        expected = expected.strip()
        for blob in (content, reasoning, f"{reasoning}\n{content}".strip()):
            hit = _numeric_in_blob(blob, expected)
            if hit:
                return hit
    if content:
        tail = _strip_answer_wrappers(_tail_line(content))
        candidate = tail if tail and (len(tail) <= 256 or not content.count("\n")) else content
        aligned = align_answer(candidate, expected)
        if aligned != candidate or len(candidate) <= 64:
            return aligned
        if _NUMERIC_ANSWER.fullmatch(expected or ""):
            return aligned
        if len(candidate) <= 256:
            return align_answer(candidate, expected)
    if reasoning:
        parsed = _json_object(reasoning)
        if parsed and parsed.get("answer") is not None:
            return align_answer(str(parsed["answer"]), expected)
        match = re.search(
            r"(?:final answer|answer)\s*[:=]\s*([^\n]+)",
            reasoning,
            flags=re.IGNORECASE,
        )
        if match:
            return align_answer(match.group(1), expected)
        tail = _strip_answer_wrappers(_tail_line(reasoning))
        if tail:
            return align_answer(tail, expected)
    return align_answer(content or reasoning, expected)


def _normalize_call(call: dict) -> dict:
    function = call.get("function", call)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"code": arguments}
    return {"name": function.get("name") or call.get("name"), "arguments": arguments}


def _reasoning_body(message: dict, parsed: dict, answer: str) -> str | None:
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    if parsed.get("reasoning"):
        return str(parsed["reasoning"]).strip()
    if parsed.get("reasoning_content"):
        return str(parsed["reasoning_content"]).strip()
    if reasoning:
        if answer and reasoning.endswith(answer):
            prefix = reasoning[: -len(answer)].strip()
            if len(prefix) > 20:
                return prefix
        return reasoning
    if content and answer and content != answer and len(content) > len(answer) + 20:
        return content
    return reasoning or None


def interpret_teacher(message: dict | str) -> dict:
    """Accept either a raw string or an OpenAI message dict."""
    if isinstance(message, str):
        message = {"content": message}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    parsed = _json_object(content) or _json_object(reasoning) or {}
    tool_calls = parsed.get("tool_calls") or message.get("tool_calls") or []
    normalized = [_normalize_call(call) for call in tool_calls]
    if normalized:
        assistant = {"role": "assistant", "content": None, "tool_calls": normalized}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        return assistant

    answer = extract_graded_answer(message, expected=None)
    reasoning_text = _reasoning_body(message, parsed, answer)
    assistant: dict = {"role": "assistant", "content": answer}
    if reasoning_text:
        assistant["reasoning_content"] = reasoning_text
    return assistant
