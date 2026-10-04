"""OpenAI-compatible chat client. Works against a remote API or a local server."""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from queue import Queue
from typing import BinaryIO, Iterator, Sequence, TextIO

logger = logging.getLogger("sftmill.teacher")


def _tool_call_for_api(call: dict, *, call_id: str) -> dict:
    if call.get("type") == "function" and isinstance(call.get("function"), dict):
        out = dict(call)
        out.setdefault("id", call_id)
        out.setdefault("type", "function")
        function = dict(out["function"])
        arguments = function.get("arguments", {})
        if isinstance(arguments, dict):
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        out["function"] = function
        return out
    name = call.get("name") or (call.get("function") or {}).get("name")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = (call.get("function") or {}).get("arguments", {})
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif arguments is None:
        arguments = "{}"
    return {
        "id": call.get("id") or call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def messages_for_chat_api(messages: list[dict]) -> list[dict]:
    """Map sftmill's stored messages onto OpenAI chat-completions shape."""
    api_messages: list[dict] = []
    pending_call_ids: list[str] = []
    tool_reply_index = 0
    call_counter = 0

    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            pending_call_ids = []
            tool_reply_index = 0
            converted = dict(message)
            api_calls = []
            for call in message["tool_calls"]:
                call_id = f"call_{call_counter}"
                call_counter += 1
                api_calls.append(_tool_call_for_api(call, call_id=call_id))
                pending_call_ids.append(api_calls[-1]["id"])
            converted["tool_calls"] = api_calls
            api_messages.append(converted)
            continue
        if role == "tool":
            converted = dict(message)
            if pending_call_ids and tool_reply_index < len(pending_call_ids):
                converted["tool_call_id"] = pending_call_ids[tool_reply_index]
                tool_reply_index += 1
            api_messages.append(converted)
            continue
        pending_call_ids = []
        tool_reply_index = 0
        api_messages.append(dict(message))
    return api_messages


def build_chat_request(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list | None = None,
    temperature: float = 0.2,
    api_key: str | None = None,
    *,
    stream: bool = False,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> tuple[str, dict, dict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages_for_chat_api(messages),
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if tools:
        payload["tools"] = tools
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return url, payload, headers


def message_from_completion(body: dict) -> dict:
    choice = body["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls"),
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
    }


def _merge_tool_call_delta(store: dict[int, dict], delta_calls: list) -> None:
    for call in delta_calls:
        index = int(call.get("index", 0))
        entry = store.setdefault(index, {"function": {"name": "", "arguments": ""}})
        if call.get("id"):
            entry["id"] = call["id"]
        if call.get("type"):
            entry["type"] = call["type"]
        function = call.get("function") or {}
        if function.get("name"):
            entry["function"]["name"] += function["name"]
        if function.get("arguments"):
            entry["function"]["arguments"] += function["arguments"]


def message_from_stream_chunks(chunks: list[dict]) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
            if delta.get("tool_calls"):
                _merge_tool_call_delta(tool_calls, delta["tool_calls"])
    message = {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts) or None,
        "tool_calls": None,
        "finish_reason": finish_reason,
    }
    if tool_calls:
        message["tool_calls"] = [
            tool_calls[index] for index in sorted(tool_calls)
        ]
    return message


def iter_sse_payloads(raw: BinaryIO) -> list[dict]:
    """Read an OpenAI-style SSE body and return parsed JSON chunks."""
    chunks: list[dict] = []
    for raw_line in raw:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line == "data: [DONE]":
            break
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            logger.debug("skip non-json sse line: %s", payload[:120])
    return chunks


def stream_chat_completion(
    url: str,
    payload: dict,
    headers: dict,
    *,
    timeout: float,
    stream_to: TextIO | None = None,
) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST",
    )
    sink = stream_to or sys.stdout
    logger.info("POST %s stream=true model=%s", url, payload.get("model"))
    chunks: list[dict] = []
    wrote = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                logger.debug("skip non-json sse line: %s", line[6:120])
                continue
            chunks.append(chunk)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    piece = delta.get(key)
                    if piece:
                        sink.write(piece)
                        sink.flush()
                        wrote = True
    message = message_from_stream_chunks(chunks)
    logger.debug(
        "stream finished content_chars=%s reasoning_chars=%s tool_calls=%s finish_reason=%s",
        len(message.get("content") or ""),
        len(message.get("reasoning_content") or ""),
        bool(message.get("tool_calls")),
        message.get("finish_reason"),
    )
    if wrote and sink is sys.stdout:
        sink.write("\n")
        sink.flush()
    return message


def _complete_non_streaming(
    url: str,
    payload: dict,
    headers: dict,
    *,
    timeout: float,
) -> dict:
    payload = dict(payload)
    payload["stream"] = False
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST",
    )
    logger.info("POST %s stream=false model=%s", url, payload.get("model"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return message_from_completion(body)


class OpenAICompatibleTeacher:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout: float = 120,
        *,
        stream: bool = True,
        stream_to: TextIO | None = None,
        max_tokens: int | None = 8192,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.stream = stream
        self.stream_to = stream_to
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[dict],
        tools: list | None = None,
        *,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        limit = self.max_tokens if max_tokens is None else max_tokens
        url, payload, headers = build_chat_request(
            self.base_url,
            self.model,
            messages,
            tools,
            self.temperature,
            self.api_key,
            stream=self.stream,
            max_tokens=limit,
            response_format=response_format,
        )
        roles = [message.get("role") for message in messages]
        logger.info(
            "teacher complete stream=%s messages=%s tools=%s",
            self.stream,
            len(messages),
            bool(tools),
        )
        logger.debug("message roles=%s", roles)
        try:
            if self.stream:
                return stream_chat_completion(
                    url, payload, headers,
                    timeout=self.timeout,
                    stream_to=self.stream_to,
                )
            return _complete_non_streaming(url, payload, headers, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            logger.error("teacher HTTP %s: %s", exc.code, detail)
            raise


class TeacherPool:
    """Fixed set of teacher slots. ``borrow()`` blocks until a slot is free."""

    def __init__(self, teachers: Sequence) -> None:
        slots = list(teachers)
        if not slots:
            raise ValueError("TeacherPool needs at least one teacher")
        self._idle: Queue = Queue()
        for teacher in slots:
            self._idle.put(teacher)
        self.size = len(slots)

    @contextmanager
    def borrow(self) -> Iterator:
        teacher = self._idle.get()
        try:
            yield teacher
        finally:
            self._idle.put(teacher)


def teacher_pool(teacher, jobs: int | None = None) -> TeacherPool:
    """Wrap a teacher, a list of teachers, or an existing pool."""
    if isinstance(teacher, TeacherPool):
        return teacher
    if isinstance(teacher, (list, tuple)):
        return TeacherPool(teacher)
    n = 1 if jobs is None else max(1, int(jobs))
    return TeacherPool([teacher] * n)
