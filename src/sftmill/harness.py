"""Harness specs are data. Alice stays on its Jinja template.

The teacher emits canonical tool names. A spec renames them only when the
training text is rendered.
"""

from __future__ import annotations

import json

CANONICAL_TOOLS = (
    "python",
    "list_dir",
    "read_file",
    "write_file",
    "edit_file",
    "search",
    "bash",
)

OPENAI_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "StrReplace",
    "search": "Grep",
    "bash": "Shell",
    "list_dir": "Glob",
    "python": "python",
}

CLAUDE_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "search": "Grep",
    "bash": "Bash",
    "list_dir": "Glob",
    "python": "python",
}

_TRAIN_ENVELOPES = (
    {
        "call_template": "<{name}>\n{arguments}\n</{name}>",
        "result_template": "OBSERVE {name}: {content}",
        "reasoning_template": "",
    },
    {
        "call_template": "[[{name}]]\n{arguments}",
        "result_template": "[[/{name}]]\n{content}",
        "reasoning_template": "{reasoning}\n",
    },
)

_HELD_OUT_ENVELOPE = {
    "call_template": "@@{name}@@\n{arguments}",
    "result_template": "@@/{name}@@\n{content}",
    "reasoning_template": "",
}

_NOVEL_NAMES = {
    "read_file": "nova_read",
    "write_file": "nova_write",
    "edit_file": "nova_edit",
    "search": "nova_find",
    "bash": "nova_run",
    "list_dir": "nova_list",
    "python": "nova_py",
}


def _tool_preface(name_map: dict, rules: str) -> str:
    lines = [rules.rstrip(), "Tools:"]
    for canonical in CANONICAL_TOOLS:
        lines.append(f"- {name_map.get(canonical, canonical)}")
    return "\n".join(lines)


def openai_spec() -> dict:
    return {
        "id": "openai",
        "name_map": dict(OPENAI_NAMES),
        "call_template": '{{"name":"{name}","arguments":{arguments}}}',
        "result_template": "tool {name}: {content}",
        "reasoning_template": "",
        "preface": _tool_preface(
            OPENAI_NAMES,
            "Call tools as one JSON object with name and arguments. "
            "Wait for the tool result before the next call. "
            "The final turn is an answer with no tool call.",
        ),
    }


def claude_code_spec() -> dict:
    return {
        "id": "claude_code",
        "name_map": dict(CLAUDE_NAMES),
        "call_template": "<tool_call><tool_name>{name}</tool_name><arguments>{arguments}</arguments></tool_call>",
        "result_template": '<tool_result name="{name}">{content}</tool_result>',
        "reasoning_template": "",
        "preface": _tool_preface(
            CLAUDE_NAMES,
            "Call a tool with tool_call, tool_name, and arguments. "
            "Wait for tool_result. The final turn has no tool call.",
        ),
    }


def sample_novel(seq: int) -> dict:
    """A training spec whose envelope is not Alice, OpenAI, or Claude Code."""
    envelope = dict(_TRAIN_ENVELOPES[seq % len(_TRAIN_ENVELOPES)])
    return {
        "id": "novel",
        "name_map": dict(_NOVEL_NAMES),
        **envelope,
        "preface": _tool_preface(
            _NOVEL_NAMES,
            "Use only the tools listed here, in the call shape shown. "
            "Wait for the observation. Do not invent a tool name.",
        ),
    }


def held_out_spec() -> dict:
    """An envelope the training sampler does not emit."""
    return {
        "id": "held_out",
        "name_map": {"edit_file": "held_edit", "read_file": "held_read", "bash": "held_run"},
        **_HELD_OUT_ENVELOPE,
        "preface": "Held-out harness. Calls use @@name@@.",
    }


def materialize_harness(harness_id: str, seq: int = 1) -> dict:
    if harness_id == "alice":
        return {"id": "alice"}
    if harness_id == "openai":
        return openai_spec()
    if harness_id == "claude_code":
        return claude_code_spec()
    if harness_id == "novel":
        return sample_novel(seq)
    raise ValueError(f"unknown harness {harness_id!r}")


def is_alice(harness) -> bool:
    if harness is None:
        return False
    if harness == "alice":
        return True
    return isinstance(harness, dict) and harness.get("id") == "alice"


def display_name(canonical: str, spec: dict) -> str:
    return (spec.get("name_map") or {}).get(canonical, canonical)


def _fill(template: str, **slots: str) -> str:
    text = template
    for key, value in slots.items():
        text = text.replace("{" + key + "}", value)
    return text


def _arguments_text(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))


def _assistant_block(message: dict, spec: dict) -> str:
    slot = spec.get("reasoning_template") or ""
    reasoning = ""
    if slot and message.get("reasoning_content"):
        reasoning = _fill(slot, reasoning=str(message["reasoning_content"]))
    calls = message.get("tool_calls") or []
    if calls:
        rendered = []
        for call in calls:
            function = call.get("function", call)
            name = display_name(function.get("name") or call.get("name") or "", spec)
            rendered.append(_fill(
                spec["call_template"],
                name=name,
                arguments=_arguments_text(function.get("arguments", {})),
            ))
        return reasoning + "".join(rendered)
    return reasoning + (message.get("content") or "")


def _message_block(message: dict, spec: dict) -> str:
    role = message["role"]
    if role == "assistant":
        return _assistant_block(message, spec)
    if role == "tool":
        return _fill(
            spec["result_template"],
            name=display_name(message.get("name") or "", spec),
            content=message.get("content") or "",
        )
    return f"{role}: {message.get('content') or ''}"


def render_spec(messages: list[dict], spec: dict, bos_token: str = "") -> tuple[str, list[tuple[int, int]]]:
    """Render a non-Alice harness. Assistant spans exclude tool observations."""
    pieces: list[str] = [bos_token or ""]
    preface = (spec.get("preface") or "").rstrip()
    if preface:
        pieces.append(preface + "\n\n")
    spans: list[tuple[int, int]] = []
    for message in messages:
        block = _message_block(message, spec)
        if not block:
            continue
        if pieces and not pieces[-1].endswith("\n\n"):
            pieces.append("\n\n")
        if message["role"] == "assistant":
            start = sum(len(part) for part in pieces)
            pieces.append(block)
            spans.append((start, start + len(block)))
        else:
            pieces.append(block)
    return "".join(pieces), spans
