"""Teacher loop: think, call a tool, observe, repeat."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile

from sftmill.generate.filters import accept, short_answer
from sftmill.generate.interpret import interpret_teacher
from sftmill.schema import PYTHON_TOOL, validate_example
from sftmill.tools.python_sandbox import run_python
from sftmill.tools.workspace import Workspace, cap_output

logger = logging.getLogger("sftmill.generate.trajectories")

_CANONICAL = {"list_dir", "read_file", "write_file", "edit_file", "search", "bash"}


def _tool_body(tool: dict) -> dict:
    return tool.get("function", tool)


def _advertised_names(tools: list[dict]) -> set[str]:
    return {_tool_body(tool).get("name") for tool in tools}


def _missing_required(tool_body: dict | None, arguments: dict) -> list[str]:
    if tool_body is None:
        return []
    required = (tool_body.get("parameters") or {}).get("required") or []
    return [key for key in required if arguments.get(key) in (None, "")]


def _arguments(call: dict) -> dict | None:
    function = call.get("function", call)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return arguments


def _call_name(call: dict) -> str:
    function = call.get("function", call)
    return function.get("name") or call.get("name") or ""


_PYTHON_SUFFIX = (
    "\n\nUse the python tool to compute the result. Print the final answer on its own line, "
    "then reply with that exact value in the answer field."
)


def _without_python_suffix(task: dict) -> dict:
    names = _advertised_names(task.get("tools") or [])
    if "python" in names:
        return task
    messages = []
    for message in task.get("messages") or []:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content.endswith(_PYTHON_SUFFIX):
            message = {**message, "content": content[: -len(_PYTHON_SUFFIX)]}
        messages.append(message)
    return {**task, "messages": messages}


def _run_call(call: dict, tools: list[dict], workspace: Workspace, sandbox, success: str = "ok") -> str:
    name = _call_name(call)
    if name not in _advertised_names(tools):
        return f"error: unknown tool {name}"
    arguments = _arguments(call)
    if arguments is None:
        return "error: invalid arguments"
    body = next((item for item in (_tool_body(tool) for tool in tools) if item.get("name") == name), None)
    missing = _missing_required(body, arguments)
    if missing:
        return "error: invalid arguments missing " + ", ".join(missing)
    if name == "python":
        return cap_output(sandbox(arguments.get("code", "")), workspace.max_output)
    if name in _CANONICAL:
        return workspace.call(name, arguments)
    return success


def _call_key(call: dict) -> str:
    arguments = _arguments(call) or {}
    return _call_name(call) + "\n" + json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _repeated_call(messages: list[dict]) -> bool:
    keys: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            keys.append(_call_key(call))
    run = 1
    for previous, current in zip(keys, keys[1:]):
        run = run + 1 if current == previous else 1
        if run >= 3:
            return True
    return False


def _failed_observation(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("error:") or "fail" in lowered


def _normalize_file_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").rstrip("\n") + "\n"


def _file_texts_equal(got: str, expected: str) -> bool:
    return _normalize_file_text(got) == _normalize_file_text(expected)


def _task_category(task_id: str) -> str:
    import re

    match = re.match(r"((?:\w+_)*\w+)-\d{4}", task_id or "")
    return match.group(1) if match else ""


def _maybe_append_workspace_hints(task: dict, messages: list[dict]) -> None:
    files = task.get("files")
    if not files or not messages:
        return
    user = messages[0]
    if user.get("role") != "user":
        return
    content = user.get("content") or ""
    paths = ", ".join(sorted(files))
    parts: list[str] = []
    if paths and paths not in content:
        parts.append(f"Seeded workspace files: {paths}.")
    category = _task_category(task.get("id", ""))
    if category == "api_rename" and "every import and call site" not in content.lower():
        parts.append(
            f"Rename across every seeded file. When done, reply with exactly {task.get('answer')}."
        )
    if not parts:
        return
    messages[0] = {**user, "content": content + "\n\n" + " ".join(parts)}


def _files_match(workspace: Workspace, expected: dict | None) -> bool:
    if not expected:
        return True
    for path, content in expected.items():
        got = workspace.read_text(str(path))
        if got is None or not _file_texts_equal(got, str(content)):
            return False
    return True


def _observation_has(messages: list[dict], needle: str | None) -> bool:
    if not needle:
        return True
    return any(
        message.get("role") == "tool" and needle in (message.get("content") or "")
        for message in messages
    )


def _tool_calls_valid(calls: list[dict]) -> bool:
    for call in calls:
        function = call.get("function") or call
        arguments = function.get("arguments", call.get("arguments", {}))
        if isinstance(arguments, dict):
            continue
        if isinstance(arguments, str) and arguments.strip():
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                return False
    return True


def _teacher_turn(teacher, messages, tools, task_id: str, step: int) -> dict | None:
    raw = teacher.complete(messages, tools)
    if isinstance(raw, str):
        raw = {"content": raw}
    if raw.get("finish_reason") == "length":
        logger.warning("trajectory rejected task=%s step=%s teacher hit max_tokens", task_id, step)
        return None
    if raw.get("tool_calls") and not _tool_calls_valid(raw["tool_calls"]):
        logger.warning("trajectory rejected task=%s step=%s incomplete tool call", task_id, step)
        return None
    return interpret_teacher(raw)


_OK_ANSWER_SUFFIX = (
    "\n\nWhen the check prints PASSED, your final reply must be exactly ok with no other text."
)


def _maybe_append_ok_hint(task: dict, messages: list[dict]) -> None:
    if str(task.get("answer", "")).strip() != "ok":
        return
    if not task.get("require_observation"):
        return
    user = messages[0]
    if user.get("role") != "user":
        return
    content = user.get("content") or ""
    if "final reply must be exactly ok" in content:
        return
    messages[0] = {**user, "content": content + _OK_ANSWER_SUFFIX}


def verify_passes(workspace: Workspace, task: dict) -> bool:
    """Re-run the task check. A printed PASSED in the transcript does not count."""
    command = str(task.get("verify") or "")
    if not command:
        return True
    code, stdout, _stderr = workspace.run(command)
    if code != 0 or stdout.startswith("error:"):
        return False
    if task.get("verify_stdout") == "exact":
        return stdout.strip() == str(task.get("answer", "")).strip()
    marker = str(task.get("require_observation") or "PASSED")
    return marker in stdout


def _reseed(workspace: Workspace, files: dict | None) -> None:
    """Grade a lookup against the seed tree, ignoring files the solver added."""
    for child in list(workspace.root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    workspace.seed(files)


def _restore_checks(workspace: Workspace, task: dict) -> None:
    """Put seed files the gold edit does not change back before the check runs."""
    files = task.get("files") or {}
    expected = task.get("expect_files") or {}
    for path, content in files.items():
        if path not in expected:
            workspace.write_file(str(path), str(content))


def generate_trajectory(task: dict, teacher, sandbox=run_python, max_steps: int = 4) -> dict | None:
    task_id = task.get("id", "?")
    limit = int(task["max_steps"]) if task.get("max_steps") is not None else max_steps
    logger.info("trajectory task=%s max_steps=%s", task_id, limit)
    task = _without_python_suffix(task)
    messages = list(task["messages"])
    _maybe_append_ok_hint(task, messages)
    _maybe_append_workspace_hints(task, messages)
    tool_success = short_answer(str(task["answer"])) if task.get("answer") is not None else "ok"
    tools = task.get("tools") or [PYTHON_TOOL]
    correction = task.get("correction")
    inserted = False
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(tmp)
        workspace.seed(task.get("files"))

        def run_sandbox(code: str) -> str:
            if sandbox is run_python:
                return run_python(code, cwd=str(workspace.root))
            return sandbox(code)

        tool_rounds = 0
        while tool_rounds < limit:
            logger.info("trajectory task=%s teacher step=%s", task_id, tool_rounds + 1)
            assistant = _teacher_turn(teacher, messages, tools, task_id, tool_rounds + 1)
            if assistant is None:
                return None
            messages.append(assistant)
            calls = assistant.get("tool_calls") or []
            if not calls:
                break
            for call in calls:
                stdout = _run_call(call, tools, workspace, run_sandbox, success=tool_success)
                logger.info("trajectory task=%s tool=%s stdout=%r", task_id, _call_name(call), stdout[:200])
                messages.append({"role": "tool", "name": _call_name(call), "content": stdout})
            tool_rounds += 1
            if correction and not inserted and any(
                message.get("role") == "tool" and _failed_observation(message.get("content") or "")
                for message in messages[-len(calls):]
            ):
                messages.append({"role": "user", "content": str(correction)})
                inserted = True
        else:
            if messages[-1]["role"] != "assistant":
                assistant = _teacher_turn(teacher, messages, tools, task_id, tool_rounds + 1)
                if assistant is None:
                    return None
                messages.append(assistant)
                if assistant.get("tool_calls"):
                    logger.warning("trajectory rejected task=%s final turn still has tool calls", task_id)
                    return None
        if messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
            logger.warning("trajectory rejected task=%s did not end on assistant", task_id)
            return None
        if _repeated_call(messages):
            logger.warning("trajectory rejected task=%s repeated tool call", task_id)
            return None
        example = {
            "kind": "trajectory",
            "task_id": task_id,
            "messages": messages,
            "tools": tools,
        }
        if task.get("harness") is not None:
            example["harness"] = task["harness"]
        if task.get("group_id"):
            example["group_id"] = task["group_id"]
        if task.get("answer") is not None:
            example["answer"] = task["answer"]
        if task.get("verify"):
            if task.get("verify_stdout") == "exact" and not task.get("expect_files"):
                _reseed(workspace, task.get("files"))
            else:
                _restore_checks(workspace, task)
            if not verify_passes(workspace, task):
                logger.warning("trajectory rejected task=%s verify failed", task_id)
                return None
        elif not _files_match(workspace, task.get("expect_files")):
            logger.warning("trajectory rejected task=%s expect_files mismatch", task_id)
            return None
        if not _observation_has(messages, task.get("require_observation")):
            logger.warning("trajectory rejected task=%s missing observation", task_id)
            return None
        if not accept(example, task):
            logger.warning("trajectory rejected task=%s answer mismatch", task_id)
            return None
        logger.info("trajectory kept task=%s", task_id)
        return validate_example(example)


def generate_trajectories(tasks: list[dict], teacher, sandbox=run_python, max_steps: int = 4):
    kept = 0
    total = sum(1 for task in tasks if task.get("kind") == "trajectory")
    for task in tasks:
        if task.get("kind") != "trajectory":
            continue
        row = generate_trajectory(task, teacher, sandbox=sandbox, max_steps=max_steps)
        if row is not None:
            kept += 1
            yield row
    logger.info("trajectories kept %s/%s", kept, total)
