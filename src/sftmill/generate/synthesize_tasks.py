"""Synthesize tasks.jsonl from a curriculum YAML via an OpenAI-compatible teacher."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging

import yaml

from sftmill.harness import materialize_harness
from sftmill.progress import ProgressReporter
from sftmill.schema import PYTHON_TOOL, WORKSPACE_TOOLS, validate_example
from sftmill.generate.trajectories import verify_passes
from sftmill.tools.workspace import Workspace

logger = logging.getLogger("sftmill.generate.tasks")

SYNTHESIS_MAX_TOKENS = 32768
MAX_REPAIR_TURNS = 2
_PASSED_CATEGORIES = {"search_edit", "regression", "long_feature"}


def load_curriculum(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "categories" not in data:
        raise ValueError("curriculum must be a mapping with a categories list")
    categories = data["categories"]
    if not categories:
        raise ValueError("curriculum needs at least one category")
    for category in categories:
        for key in ("id", "count", "kind", "match", "template"):
            if key not in category:
                raise ValueError(f"category {category.get('id', '?')} missing {key}")
        if category["kind"] not in {"trace", "trajectory"}:
            raise ValueError(f"category {category['id']} has invalid kind {category['kind']!r}")
        if category["match"] not in {"exact", "contains", "open"}:
            raise ValueError(f"category {category['id']} has invalid match {category['match']!r}")
        if int(category["count"]) < 1:
            raise ValueError(f"category {category['id']} count must be positive")
        if "max_steps" in category:
            steps = category["max_steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError(f"category {category['id']} max_steps must be a positive integer")
        if "harnesses" in category:
            allowed = {"alice", "openai", "claude_code", "novel"}
            unknown = [name for name in category["harnesses"] if name not in allowed]
            if unknown:
                raise ValueError(f"category {category['id']} has unknown harnesses {unknown}")
        if "toolset" in category and category["toolset"] not in {"workspace", "custom"}:
            raise ValueError(f"category {category['id']} has invalid toolset {category['toolset']!r}")
        if "verify_on" in category and category["verify_on"] != "seed":
            raise ValueError(f"category {category['id']} verify_on must be seed")
        if "user_turns" in category:
            turns = category["user_turns"]
            if isinstance(turns, bool) or not isinstance(turns, int) or turns not in {2, 3}:
                raise ValueError(f"category {category['id']} user_turns must be 2 or 3")
    return data


def _extract_json_array(text: str) -> list | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            return value
    objects = _json_objects(cleaned)
    return objects or None


def _json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    found: list[dict] = []
    index = text.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("{", index + 1)
            continue
        if isinstance(value, dict) and (value.get("question") or value.get("prompt") or value.get("user")):
            found.append(value)
        index = text.find("{", end)
    return found


def _extract_json_object(text: str) -> dict | None:
    if not text or not str(text).strip():
        return None
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value[0]
    except json.JSONDecodeError:
        pass
    objects = _json_objects(cleaned)
    return objects[-1] if objects else None


def category_response_schema(category: dict) -> dict:
    if category.get("match") == "open":
        if category.get("user_turns"):
            properties = {"turns": {"type": "array", "items": {"type": "string"}}}
            required = ["turns"]
        else:
            properties = {"question": {"type": "string"}}
            required = ["question"]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": category["id"],
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": True,
                },
            },
        }
    properties: dict = {
        "question": {"type": "string"},
        "answer": {"type": "string"},
    }
    required = ["question", "answer"]
    if category.get("toolset") == "workspace":
        file_map = {"type": "object", "additionalProperties": {"type": "string"}}
        properties["files"] = file_map
        required.append("files")
        if category.get("verify_on") == "seed":
            properties["verify"] = {"type": "string"}
            required.append("verify")
        else:
            properties["expect_files"] = file_map
            required.append("expect_files")
            if category["id"] == "long_feature":
                properties["correction"] = {"type": "string"}
                required.append("correction")
    elif category.get("toolset") == "custom":
        properties["tools"] = {"type": "array", "items": {"type": "object"}}
        required.append("tools")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": category["id"],
            "strict": False,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": True,
            },
        },
    }


def _call_synthesis_teacher(teacher, messages: list[dict], response_format: dict) -> dict:
    try:
        return teacher.complete(
            messages,
            response_format=response_format,
            max_tokens=SYNTHESIS_MAX_TOKENS,
        )
    except urllib.error.HTTPError:
        logger.warning("task synthesis retry without response_format")
        return teacher.complete(messages, max_tokens=SYNTHESIS_MAX_TOKENS)


def _teacher_text(raw) -> tuple[str, str]:
    if isinstance(raw, str):
        return raw, ""
    return raw.get("content") or "", raw.get("reasoning_content") or ""


def _parse_teacher_object(raw) -> dict | None:
    content, reasoning = _teacher_text(raw)
    return _extract_json_object(content) or _extract_json_object(reasoning)


def _file_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        text = value.get("content") or value.get("text") or value.get("body")
        if isinstance(text, str) and text.strip():
            return text
    return None


def _coerce_files(value: object) -> dict[str, str] | None:
    pairs: list[tuple[object, object]] = []
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path") or entry.get("name") or entry.get("file")
            body = entry.get("content", entry.get("text", entry.get("body")))
            pairs.append((path, body))
    files: dict[str, str] = {}
    for path, body in pairs:
        text = _file_text(body)
        if isinstance(path, str) and path.strip() and text is not None:
            files[path] = text
    return files or None


def _coerce_turns(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    turns: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict) and entry.get("role", "user") == "user":
            text = str(entry.get("content") or "").strip()
        else:
            return None
        if not text:
            return None
        turns.append(text)
    return turns


def _normalize_item(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    question = raw.get("question") or raw.get("prompt") or raw.get("user")
    answer = raw.get("answer") or raw.get("expected_answer")
    turns = _coerce_turns(raw.get("turns")) if "turns" in raw else None
    if "turns" in raw and turns is None:
        return {"question": str(question or "turns").strip() or "turns", "turns": []}
    if turns:
        question = "\n".join(turns)
        item = {"question": question, "turns": turns}
        if answer is not None and str(answer).strip():
            item["answer"] = str(answer).strip()
        for key in ("files", "expect_files"):
            files = _coerce_files(raw.get(key))
            if files:
                item[key] = files
        for key in ("tools", "require_observation", "correction", "verify"):
            if raw.get(key):
                item[key] = raw[key]
        return item
    if question is None or answer is None:
        if question is None or not str(question).strip():
            return None
        return {"question": str(question).strip()}
    question = str(question).strip()
    answer = str(answer).strip()
    if not question or not answer:
        return None
    item = {"question": question, "answer": answer}
    for key in ("files", "expect_files"):
        files = _coerce_files(raw.get(key))
        if files:
            item[key] = files
    for key in ("tools", "require_observation", "correction", "verify"):
        if raw.get(key):
            item[key] = raw[key]
    return item


_WORKSPACE_NAMES = {"python", "list_dir", "read_file", "write_file", "edit_file", "search", "bash"}


def _text_files(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(path, str) and isinstance(body, str) and body for path, body in value.items())
    )


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _named_answer(category_id: str, question: str, answer: str) -> str | None:
    if category_id not in {"design_owner", "review_finding"}:
        return None
    if len(answer.split()) != 1 or not _has_token(question, answer):
        return "answer must be one token named in the question"
    if category_id == "review_finding" and re.search(
        rf"\bthe (bug|fault) is\s+{re.escape(answer)}\b",
        question,
        flags=re.IGNORECASE,
    ):
        return "question states the fault"
    return None


def _reject_open(category: dict, item: dict) -> str | None:
    if item.get("answer"):
        return "open tasks have no gold answer"
    expected = category.get("user_turns")
    if expected:
        turns = item.get("turns") or []
        if len(turns) != expected:
            return f"turns must be {expected} user messages"
        return None
    if item.get("turns"):
        return "single-turn tasks use question, not turns"
    if not str(item.get("question") or "").strip():
        return "missing question"
    return None


def _reject_reason(category: dict, item: dict) -> str | None:
    if category.get("match") == "open":
        return _reject_open(category, item)
    if not item.get("answer"):
        return "missing answer"
    question = item["question"]
    answer = item["answer"]
    if category["id"] == "bug_needle":
        if answer in question or len(answer.split()) != 1:
            return "needle leaked or answer is not one token"
    named = _named_answer(category["id"], question, answer)
    if named:
        return named
    if category.get("toolset") == "custom":
        tools = item.get("tools")
        if not isinstance(tools, list) or not tools:
            return "missing tools"
        names = set()
        for tool in tools:
            if isinstance(tool, dict):
                names.add((tool.get("function") or tool).get("name"))
        if names & _WORKSPACE_NAMES:
            return "custom tools used a workspace name"
        if "\n" in answer or len(answer) > 40 or "Trajectory:" in answer or "Result:" in answer:
            return "answer is not a short value"
    if category.get("toolset") == "workspace" and category.get("verify_on") == "seed":
        return _reject_seed_lookup(category, item)
    if category.get("toolset") == "workspace":
        if not _text_files(item.get("files")) or not _text_files(item.get("expect_files")):
            return "missing files or expect_files"
        if item.get("expect_files") == item.get("files"):
            return "expect_files matches files"
        bodies = list(item["files"].values()) + list(item["expect_files"].values())
        if category.get("require_observation") or category["id"] in _PASSED_CATEGORIES:
            if not any("PASSED" in body for body in bodies):
                return "no file prints PASSED"
        if category["id"] in {"search_edit", "boundary_repair", "incident_fix"} and len(item["files"]) < 3:
            return "files need at least three paths"
        for body in item["files"].values():
            if len(body) >= 40 and body in question:
                return "question contains a file body"
        if category["id"] == "long_feature" and not str(item.get("correction") or "").strip():
            return "missing correction"
        if category["id"] == "long_feature":
            correction = str(item.get("correction") or "")
            if correction and correction in question:
                return "correction text is in the question"
            reason = _reject_long_feature_correction(item, correction)
            if reason:
                return reason
        if category["id"] == "api_rename":
            if answer not in question:
                return "question does not name the new identifier"
            if answer not in "".join(item["expect_files"].values()):
                return "new name missing from expect_files"
        if category["id"] == "cli_repair":
            if "check.py" not in item["files"] or "check.py" in item["expect_files"]:
                return "check.py must stay in the seed and out of expect_files"
        proof = _proof_failure(category, item)
        if proof:
            return proof
    return None


def _reject_seed_lookup(category: dict, item: dict) -> str | None:
    """A hidden command on the seed tree must print the answer."""
    question = item["question"]
    answer = item["answer"]
    files = item.get("files")
    if not _text_files(files) or len(files) < 2:
        return "missing files"
    if item.get("expect_files"):
        return "expect_files is not used"
    verify = str(item.get("verify") or "").strip()
    if not verify:
        return "missing verify"
    if "&&" in verify or ";" in verify or ".." in verify:
        return "verify command is not a single workspace command"
    if _answer_mentioned(question, answer) or verify in question:
        return "answer or verify command is in the question"
    if _answer_mentioned(verify, answer):
        return "answer is inside the verify command"
    if not any(path in verify for path in files):
        return "verify command does not read a seeded file"
    if "\n" in answer or len(answer) > 80:
        return "answer is not one short line"
    for body in files.values():
        if len(body) >= 40 and body in question:
            return "question contains a file body"
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(tmp)
        workspace.seed(files)
        code, stdout, _stderr = workspace.run(verify)
        if stdout.startswith("error:"):
            return "verify command failed"
        if stdout.strip() != answer:
            return "verify stdout does not match answer"
        if code != 0:
            return "verify command must exit 0"
    return None


def _number_key(raw: str) -> str:
    if "." not in raw:
        return str(int(raw))
    whole, frac = raw.split(".", 1)
    frac = frac.rstrip("0")
    if not frac:
        return str(int(whole))
    return f"{int(whole)}.{frac}"


def _number_keys(text: str) -> set[str]:
    return set(_numbers_in_order(text))


def _numbers_in_order(text: str) -> list[str]:
    return [_number_key(raw) for raw in re.findall(r"\d+(?:\.\d+)?", text)]


def _reject_long_feature_correction(item: dict, correction: str) -> str | None:
    """The correction must say to use the gold value, then name a wrong value that is not the seed."""
    seed_program = "\n".join(body for path, body in item["files"].items() if path != "check.py")
    gold = _number_keys("\n".join(item["expect_files"].values()))
    introduced = gold - _number_keys(seed_program)
    ordered = _numbers_in_order(correction)
    mentioned = set(ordered)
    if not introduced or not (mentioned & introduced):
        return "correction must name the new value in expect_files"
    everywhere = _number_keys("\n".join(item["files"].values())) | gold
    if not (mentioned - everywhere):
        return "correction must name a wrong value that is not the seed"
    first_gold = next(index for index, number in enumerate(ordered) if number in introduced)
    if any(index < first_gold for index, number in enumerate(ordered) if number not in everywhere):
        return "correction must set the gold value, not the wrong one"
    return None


def _answer_mentioned(text: str, answer: str) -> bool:
    if re.fullmatch(r"\w+", answer):
        return _has_token(text, answer)
    return answer in text


def _proof_failure(category: dict, item: dict) -> str | None:
    """The check must fail on the seed and pass after the gold edit."""
    command = category.get("verify") or item.get("verify")
    if not command:
        return None
    task = {
        "verify": command,
        "answer": item.get("answer"),
        "verify_stdout": category.get("verify_stdout") or item.get("verify_stdout"),
        "require_observation": category.get("require_observation") or item.get("require_observation"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(tmp)
        workspace.seed(item.get("files"))
        if verify_passes(workspace, task):
            return "verify already passes on the seed"
        workspace.seed(item.get("expect_files"))
        if not verify_passes(workspace, task):
            return "verify fails on expect_files"
    return None


def _trajectory_user_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Use the python tool to compute the result. Print the final answer on its own line, "
        "then reply with that exact value in the answer field."
    )


def _question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


_CATEGORY_CHECKS = {
    "bug_needle": (
        "The snippet and the call in the question raise exactly answer. "
        "Do not name that token in the question."
    ),
    "incident_fix": (
        "The log line's expected value is the result of the gold edit. "
        "The assertion that already passes is a case the report says is already correct."
    ),
    "contract_feature": (
        "Hardcoding the single sample's formatted string must fail the check."
    ),
    "design_owner": (
        "Exactly one named component can own the responsibility. "
        "If a second candidate also fits the constraints, rewrite the question until it does not."
    ),
    "review_finding": (
        "answer is the function that breaks the stated rule. "
        "The other function must satisfy that rule."
    ),
    "python_repair": (
        "Each fault is a broken standard formula: a wrong operator, a missing factor, or swapped arguments. "
        "Do not retune a rate, fee, or constant unless the question states the correct value. "
        "answer is the printed line those corrected formulas produce."
    ),
    "test_first": (
        "test_new.py is absent from files and present in expect_files. "
        "test_contract asserts the behavior the question describes, and that assertion fails on the seed."
    ),
    "compat_change": (
        "The question states the rule for each call style, including the result or formula it must produce. "
        "'Both must keep working' is not a rule. The gold edit implements that stated rule. "
        "The two styles require different results. One style already passes on the seed, "
        "and the gold edit must keep that style working."
    ),
    "long_feature": (
        "correction is one sentence: set the setting to the gold value, not a wrong value. "
        "The gold value is the new number in expect_files, and it is the number the check accepts. "
        "The wrong value is neither the seed nor the gold, and using it fails the check. "
        "Do not tell the solver to apply the wrong value. Do not mention the number already in the seed. "
        "When the corrected setting is a rate or coefficient, do not write the gold number as a literal in check.py. "
        "Assert the end result. The question does not state the corrected number."
    ),
    "boundary_repair": (
        "The gold edit moves the rule into the layer the question names. "
        "The check fails if that rule text is still in the wrong file, "
        "and one neighbor keeps its original result."
    ),
    "cli_find": (
        "verify is a hidden grader. It does not appear in the question, and answer appears in neither. "
        "The command names a seeded file, prints answer and nothing else, and exits 0. "
        "grep -c exits 1 when the count is 0. One command, no &&, no semicolon, and no '..'. "
        "The tree needs a decoy so the value is not the only number present."
    ),
    "cli_repair": (
        "The question states what the script must print or do. "
        "The fault is a wrong operator, a missing argument, or a bad pipe. "
        "Do not retune a constant unless the question states the correct value. "
        "check.py runs bash on the script and is not in expect_files."
    ),
}


def _system_prompt(category: dict) -> str:
    """Instructions that survive a repair turn. The repair user message replaces the first user turn."""
    parts = [
        "You write one training task for a sftmillation pipeline.",
        "Reply with one JSON object only, with no markdown fence and no other text.",
        "The example shows the JSON shape. Use a new domain, new names, and new numbers.",
    ]
    if category.get("match") != "open":
        parts.append(
            "File bodies go in files, never in question."
            if category.get("verify_on") == "seed"
            else "File bodies go in files and expect_files, never in question."
        )
    if category.get("match") == "exact":
        parts.append(
            "answer is the solver's entire final message: one line, "
            "with no wrapper such as 'the answer is'."
        )
    if category.get("match") == "open":
        parts.append(
            "This is a conversation task. Do not include an answer field. "
            "Another model writes the reply later."
        )
        if category.get("user_turns"):
            parts.append(
                f"turns is exactly {int(category['user_turns'])} user messages, in order, "
                "and no assistant messages. Each turn follows from the one before it."
            )
    command = category.get("verify")
    if command:
        parts.append(
            f"A checker runs `{command}` on files, then again after overlaying expect_files. "
            "The seed must fail and the gold edit must pass. Passing that check is not enough. "
            "The question, the seed's failure, and the gold's success must be the same bug: "
            "every number or outcome the question states is what the gold code produces, "
            "and a case that already passes on the seed must still match the question after the fix. "
            "Do not fix a bug by changing who the rule applies to. "
            "A check that one hardcoded string or one constant can satisfy is invalid; "
            "when the contract is a format or a formula, assert at least two different inputs. "
            f"The command in the question is `{command}`. Use python3, not python. "
            "Do not join commands with &&."
        )
    extra = _CATEGORY_CHECKS.get(category["id"])
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _build_task_messages(category: dict, existing_questions: list[str]) -> list[dict]:
    avoid = ""
    if existing_questions:
        avoid = (
            "\nThese questions are already used in this category. "
            "Write a different task. Do not repeat or paraphrase any of them:\n"
            + "\n".join(f"- {q}" for q in existing_questions)
        )
    example_block = ""
    example = category.get("example")
    if isinstance(example, dict):
        example_block = (
            "Valid example (match this shape, write a new task):\n"
            + json.dumps(example, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    system = _system_prompt(category)
    user = (
        f"Category: {category['id']}\n"
        f"Kind: {category['kind']}\n"
        f"Match mode: {category['match']}\n\n"
        f"{example_block}"
        f"{category['template'].strip()}"
        f"{avoid}\n\n"
        "Return one new JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _synthesize_one_item(
    category: dict,
    teacher,
    existing_questions: list[str],
    seen: set[str],
) -> dict | None:
    response_format = category_response_schema(category)
    messages = _build_task_messages(category, existing_questions)
    for repair in range(MAX_REPAIR_TURNS + 1):
        raw = _call_synthesis_teacher(teacher, messages, response_format)
        obj = _parse_teacher_object(raw)
        normalized = _normalize_item(obj) if obj else None
        if normalized is None:
            reason = "invalid or missing question and answer"
        else:
            reason = _reject_reason(category, normalized)
            if reason is None:
                if _question_key(normalized["question"]) in seen:
                    reason = "duplicate question"
                else:
                    return normalized
        logger.warning("category=%s repair=%s rejected: %s", category["id"], repair, reason)
        if repair >= MAX_REPAIR_TURNS:
            return None
        payload = json.dumps(obj, ensure_ascii=False) if obj else "{}"
        follow = "Return one corrected JSON object only."
        if category.get("verify"):
            follow += (
                " The question, the seed, and expect_files must describe the same bug."
                " Do not loosen the check so a wrong edit passes."
            )
        messages = messages + [
            {"role": "assistant", "content": payload},
            {"role": "user", "content": f"Rejected: {reason}. {follow}"},
        ]
    return None


def _build_meta_prompt(category: dict, batch_size: int, existing_questions: list[str]) -> list[dict]:
    """Backward-compatible alias for tests."""
    return _build_task_messages(category, existing_questions)


_EXACT_FINAL = "\n\nYour final reply must be only the answer value, with no other text."


def _user_prompt(category: dict, question: str) -> str:
    if category.get("verify_on") == "seed":
        text = (
            f"{question}\n\n"
            "Use the listed tools to inspect the workspace. Do not modify files."
        )
    elif category.get("toolset") == "workspace":
        text = (
            f"{question}\n\n"
            "Use the listed tools. Read and edit the workspace, re-run after each fix, "
            "and do not paste a one-shot script that ignores the files."
        )
    elif category.get("toolset") == "custom":
        text = (
            f"{question}\n\n"
            "Use only the listed tools. A bad call comes back as an error. "
            "Fix the arguments and call again. Reply with the exact final value."
        )
    elif category["kind"] == "trajectory":
        text = _trajectory_user_prompt(question)
    else:
        text = question
    if category.get("match") == "exact":
        text += _EXACT_FINAL
    return text


def _task_from_item(category: dict, item: dict, seq: int, harness_id: str | None = None) -> dict:
    kind = category["kind"]
    base_id = f"{category['id']}-{seq:04d}"
    if item.get("turns"):
        messages = [{"role": "user", "content": turn} for turn in item["turns"]]
    else:
        messages = [{"role": "user", "content": _user_prompt(category, item["question"])}]
    task = {
        "id": base_id,
        "group_id": base_id,
        "kind": kind,
        "match": category["match"],
        "messages": messages,
    }
    if item.get("answer"):
        task["answer"] = item["answer"]
    if item.get("turns"):
        task["turns"] = list(item["turns"])
    if "max_steps" in category:
        task["max_steps"] = int(category["max_steps"])
    if item.get("require_observation"):
        task["require_observation"] = item["require_observation"]
    elif category.get("require_observation"):
        task["require_observation"] = category["require_observation"]
    for key in ("verify", "verify_stdout"):
        if category.get(key):
            task[key] = category[key]
        elif item.get(key):
            task[key] = item[key]
    for key in ("files", "expect_files", "correction"):
        if item.get(key):
            task[key] = item[key]
    if kind == "trajectory":
        if item.get("tools"):
            task["tools"] = item["tools"]
        elif category.get("toolset") == "workspace":
            task["tools"] = list(WORKSPACE_TOOLS)
        else:
            task["tools"] = [PYTHON_TOOL]
    if harness_id:
        task["harness"] = materialize_harness(harness_id, seq)
        task["id"] = f"{task['id']}-{harness_id}"
    return task


def _expected_row_count(categories: list[dict]) -> tuple[int, int]:
    """Return (base task target, expanded row target)."""
    base = sum(int(category["count"]) for category in categories)
    expanded = 0
    for category in categories:
        mult = len(category["harnesses"]) if category.get("harnesses") else 1
        expanded += int(category["count"]) * mult
    return base, expanded


def synthesize_category_tasks(
    category: dict,
    teacher,
    *,
    batch_size: int = 5,
    max_attempts: int = 8,
    seen_questions: set[str] | None = None,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict], str | None]:
    """Fill one curriculum category up to ``count`` tasks."""
    del batch_size  # one task per synthesis call
    target = int(category["count"])
    seen = seen_questions if seen_questions is not None else set()
    rows: list[dict] = []
    asked: list[str] = []
    seq = 1
    attempt_limit = max(max_attempts, target * 6)
    attempts = 0
    if progress is not None:
        progress.set(active_category=category["id"])
    while len(rows) < target and attempts < attempt_limit:
        attempts += 1
        logger.info(
            "category=%s attempt=%s have=%s need=%s",
            category["id"], attempts, len(rows), target - len(rows),
        )
        item = _synthesize_one_item(category, teacher, asked, seen)
        if item is None:
            continue
        task = _task_from_item(category, item, seq)
        validate_example(task)
        rows.append(task)
        question = item["question"].strip()
        asked.append(question)
        seen.add(_question_key(question))
        seq += 1
        logger.info("category=%s added task %s", category["id"], task["id"])
        if progress is not None:
            progress.bump("base_tasks_done")
            progress.set(
                active_category=category["id"],
                category_have=len(rows),
                category_target=target,
            )
    expanded = _expand_harnesses(category, rows)
    if progress is not None:
        progress.bump("categories_done")
        progress.set(active_category="-")
    if len(rows) < target:
        msg = f"category {category['id']}: got {len(rows)}/{target} tasks after {attempts} attempts"
        logger.error(msg)
        return expanded, msg
    return expanded, None


def _expand_harnesses(category: dict, rows: list[dict]) -> list[dict]:
    harnesses = category.get("harnesses")
    if not harnesses:
        return rows
    expanded: list[dict] = []
    for index, row in enumerate(rows):
        for harness_id in harnesses:
            clone = dict(row)
            clone["harness"] = materialize_harness(harness_id, index + 1)
            clone["group_id"] = row.get("group_id") or row["id"]
            clone["id"] = f"{row['id']}-{harness_id}"
            expanded.append(clone)
    return expanded


class _LockedSeen:
    """Shared question set for categories running on several threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inner: set[str] = set()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._inner

    def __iter__(self):
        with self._lock:
            return iter(list(self._inner))

    def add(self, key: str) -> None:
        with self._lock:
            self._inner.add(key)


def synthesize_tasks(
    curriculum_path: str | Path,
    teacher,
    out_path: str | Path,
    *,
    batch_size: int = 5,
    jobs: int = 3,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict], list[str]]:
    curriculum = load_curriculum(curriculum_path)
    categories = curriculum["categories"]
    workers = max(1, int(jobs))
    base_target, row_target = _expected_row_count(categories)
    if progress is not None:
        progress.set(
            categories_total=len(categories),
            categories_done=0,
            base_tasks_target=base_target,
            base_tasks_done=0,
            rows_target=row_target,
            rows_written=0,
        )
    logger.info(
        "curriculum=%s categories=%s jobs=%s",
        curriculum.get("name"), len(categories), workers,
    )
    seen = _LockedSeen()
    out_path = Path(out_path)
    out_path.write_text("")
    write_lock = threading.Lock()
    shortfalls: list[str] = []
    all_rows: list[dict] = []
    rows_written_count = 0

    def append_rows(rows: list[dict]) -> None:
        nonlocal rows_written_count
        if not rows:
            return
        with write_lock:
            with out_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written_count += len(rows)
            if progress is not None:
                progress.set(rows_written=rows_written_count)

    def run(category: dict) -> list[dict]:
        logger.info("synthesizing category=%s count=%s kind=%s", category["id"], category["count"], category["kind"])
        rows, err = synthesize_category_tasks(
            category,
            teacher,
            batch_size=batch_size,
            seen_questions=seen,
            progress=progress,
        )
        append_rows(rows)
        if err:
            with write_lock:
                shortfalls.append(err)
        logger.info("category=%s done rows=%s", category["id"], len(rows))
        return rows

    if workers == 1 or len(categories) == 1:
        for category in categories:
            all_rows.extend(run(category))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(categories))) as pool:
            for rows in pool.map(run, categories):
                all_rows.extend(rows)
    logger.info("wrote %s tasks to %s", len(all_rows), out_path)
    return all_rows, shortfalls
