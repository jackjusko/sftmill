"""Training rows are OpenAI-style messages, the shape Alice was pretrained on."""

from __future__ import annotations

import json
from pathlib import Path

ROLES = {"system", "user", "assistant", "tool", "meta"}
KINDS = {"trace", "trajectory"}

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _string_prop() -> dict:
    return {"type": "string"}


PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run a Python snippet and return stdout.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

WORKSPACE_TOOLS = [
    PYTHON_TOOL,
    _tool("list_dir", "List names in a workspace directory.", {"path": _string_prop()}, ["path"]),
    _tool(
        "read_file",
        "Read a workspace file. offset and limit page the lines.",
        {"path": _string_prop(), "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["path"],
    ),
    _tool(
        "write_file",
        "Create or replace a workspace file.",
        {"path": _string_prop(), "content": _string_prop()},
        ["path", "content"],
    ),
    _tool(
        "edit_file",
        "Replace the single occurrence of old with new.",
        {"path": _string_prop(), "old": _string_prop(), "new": _string_prop()},
        ["path", "old", "new"],
    ),
    _tool(
        "search",
        "Literal search. Returns path:line:text.",
        {"pattern": _string_prop(), "path": _string_prop()},
        ["pattern"],
    ),
    _tool("bash", "Run one command in the workspace.", {"command": _string_prop()}, ["command"]),
]


def load_jsonl(path: str | Path) -> list[dict]:
    from sftmill.dataset import iter_jsonl_dataset

    return list(iter_jsonl_dataset(path))


def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def validate_message(message: dict) -> None:
    role = message.get("role")
    if role not in ROLES:
        raise ValueError(f"unsupported role {role!r}")
    if role == "tool" and not message.get("name"):
        raise ValueError("tool messages need a name")
    if role != "assistant" and message.get("content") is None and role != "tool":
        raise ValueError(f"{role} messages need content")


def validate_example(example: dict) -> dict:
    kind = example.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    messages = example.get("messages")
    if not messages:
        raise ValueError("example needs messages")
    for message in messages:
        validate_message(message)
    if kind == "trajectory" and not example.get("tools"):
        raise ValueError("trajectory examples need tools")
    return example


def assistant_message(
    content: str = "",
    reasoning: str | None = None,
    tool_calls: list | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return message
