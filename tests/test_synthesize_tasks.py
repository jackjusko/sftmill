import json
import threading

import pytest

from sftmill.generate.synthesize_tasks import (
    SYNTHESIS_MAX_TOKENS,
    _extract_json_array,
    _normalize_item,
    _reject_reason,
    _task_from_item,
    category_response_schema,
    load_curriculum,
    synthesize_category_tasks,
    synthesize_tasks,
)
from sftmill.generate.traces import generate_traces
from sftmill.generate.trajectories import generate_trajectories
from sftmill.schema import validate_example
from sftmill.teachers.openai_compat import build_chat_request


def test_load_curriculum_code_agent_test():
    data = load_curriculum("configs/curriculum/code_agent_test.yaml")
    assert data["name"] == "code_agent_test"
    assert len(data["categories"]) >= 1


def test_load_curriculum_code_agent():
    data = load_curriculum("configs/curriculum/code_agent.yaml")
    by_id = {category["id"]: category for category in data["categories"]}
    assert set(by_id) == {
        "bug_needle", "harness_recover", "search_edit", "regression",
        "api_rename", "python_repair", "long_feature",
        "design_owner", "review_finding", "boundary_repair", "contract_feature",
        "incident_fix", "test_first", "compat_change",
        "cli_find", "cli_repair",
    }
    for category in data["categories"]:
        assert isinstance(category.get("example"), dict)
    assert by_id["bug_needle"]["kind"] == "trace"
    assert by_id["bug_needle"]["match"] == "contains"
    recover = by_id["harness_recover"]
    assert recover["match"] == "exact"
    assert recover["max_steps"] == 12
    assert recover["toolset"] == "custom"
    for name in ("Read", "edit_file", "Bash"):
        assert name not in recover.get("tools", [])
    assert by_id["search_edit"]["max_steps"] == 16
    assert by_id["regression"]["max_steps"] == 48
    assert by_id["api_rename"]["max_steps"] == 32
    assert by_id["python_repair"]["max_steps"] == 32
    assert by_id["long_feature"]["max_steps"] == 48
    for key in (
        "search_edit", "regression", "api_rename", "python_repair", "long_feature",
        "harness_recover", "boundary_repair", "contract_feature", "incident_fix",
        "test_first", "compat_change", "cli_find", "cli_repair",
    ):
        assert by_id[key]["harnesses"] == ["alice", "openai", "claude_code", "novel"]
    assert "harnesses" not in by_id["design_owner"]
    assert "harnesses" not in by_id["review_finding"]


def test_max_steps_must_be_a_positive_integer(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
name: bad
categories:
  - id: math
    count: 1
    kind: trace
    match: exact
    max_steps: 0
    template: math
""",
    )
    with pytest.raises(ValueError, match="max_steps"):
        load_curriculum(path)


def test_task_from_item_copies_max_steps_and_novel_harness():
    from sftmill.harness import claude_code_spec, openai_spec

    category = {
        "id": "search_edit",
        "kind": "trajectory",
        "match": "exact",
        "max_steps": 16,
        "toolset": "workspace",
        "template": "",
    }
    copied = _task_from_item(category, {"question": "fix", "answer": "ok"}, 1, harness_id="novel")
    omitted = _task_from_item(
        {"id": "math", "kind": "trace", "match": "exact", "template": ""},
        {"question": "2+2", "answer": "4"},
        1,
    )
    assert copied["max_steps"] == 16
    assert copied["group_id"] == "search_edit-0001"
    assert copied["id"] == "search_edit-0001-novel"
    assert copied["harness"]["call_template"] not in {
        openai_spec()["call_template"],
        claude_code_spec()["call_template"],
    }
    assert "[TOOL_CALL_START]" not in copied["harness"]["call_template"]
    assert "max_steps" not in omitted


def test_files_listed_as_objects_are_kept():
    raw = {
        "question": "The check fails.",
        "answer": "ok",
        "files": [{"path": "app.py", "content": "print(1)\n"}],
        "expect_files": {"app.py": {"content": "print(2)\n"}},
    }
    item = _normalize_item(raw)
    assert item["files"] == {"app.py": "print(1)\n"}
    assert item["expect_files"] == {"app.py": "print(2)\n"}


def test_curriculum_examples_fail_then_pass():
    from sftmill.generate.synthesize_tasks import _proof_failure

    data = load_curriculum("configs/curriculum/code_agent.yaml")
    for category in data["categories"]:
        if not category.get("verify"):
            continue
        assert _proof_failure(category, category["example"]) is None


def test_long_feature_correction_must_name_the_gold_value():
    data = load_curriculum("configs/curriculum/code_agent.yaml")
    category = next(item for item in data["categories"] if item["id"] == "long_feature")
    assert _reject_reason(category, category["example"]) is None
    swapped = dict(category["example"])
    swapped["correction"] = "Set the default timeout to 10, not 0."
    assert _reject_reason(category, swapped) == "correction must name the new value in expect_files"
    seed_as_wrong = dict(category["example"])
    seed_as_wrong["correction"] = "Set the default timeout to 30, not 0."
    assert _reject_reason(category, seed_as_wrong) == "correction must name a wrong value that is not the seed"
    reversed_order = dict(category["example"])
    reversed_order["correction"] = "Set the default timeout to 10, not 30."
    assert _reject_reason(category, reversed_order) == "correction must set the gold value, not the wrong one"


def test_cli_find_example_matches_the_hidden_command():
    from sftmill.generate.synthesize_tasks import _reject_reason, _user_prompt

    data = load_curriculum("configs/curriculum/code_agent.yaml")
    category = next(item for item in data["categories"] if item["id"] == "cli_find")
    assert _reject_reason(category, category["example"]) is None
    leaked = dict(category["example"])
    leaked["question"] = "How many ERROR lines are in logs/app.log? The count is 3."
    assert _reject_reason(category, leaked) == "answer or verify command is in the question"
    solver = _user_prompt(category, category["example"]["question"])
    assert "Do not modify files" in solver
    echoed = dict(category["example"])
    echoed["verify"] = "echo 3"
    assert _reject_reason(category, echoed) == "answer is inside the verify command"
    missed = dict(category["example"])
    missed["files"] = {
        "logs/empty.log": "INFO quiet\n",
        "logs/other.log": "ERROR noise\n",
    }
    missed["verify"] = "grep -c ERROR logs/empty.log"
    missed["answer"] = "0"
    missed["question"] = "How many ERROR lines are in logs/empty.log?"
    assert _reject_reason(category, missed) == "verify command must exit 0"
    loosened = next(item for item in data["categories"] if item["id"] == "cli_repair")
    bad_gold = dict(loosened["example"])
    bad_gold["expect_files"] = dict(loosened["example"]["expect_files"])
    bad_gold["expect_files"]["check.py"] = "print('PASSED')\n"
    assert _reject_reason(loosened, bad_gold) == "check.py must stay in the seed and out of expect_files"
    edit = _user_prompt(
        {"id": "cli_repair", "kind": "trajectory", "match": "exact", "toolset": "workspace"},
        "Run python3 check.py.",
    )
    assert "Do not modify files" not in edit


def test_named_answer_is_a_whole_token_and_the_fault_is_not_stated():
    owner = {"id": "design_owner", "kind": "trace", "match": "exact"}
    leaked = {"question": "Tax belongs in Pricing.", "answer": "Pricing"}
    assert _reject_reason(owner, leaked) is None
    substring = {"question": "Use BorderStore for tax.", "answer": "Orders"}
    assert _reject_reason(owner, substring) == "answer must be one token named in the question"
    review = {"id": "review_finding", "kind": "trace", "match": "exact"}
    stated = {"question": "def save(order):\n    return order\nThe bug is save.", "answer": "save"}
    assert _reject_reason(review, stated) == "question states the fault"


def test_design_owner_final_must_be_the_token_alone():
    task = {
        "id": "design_owner-0001",
        "kind": "trace",
        "match": "exact",
        "answer": "Pricing",
        "messages": [{"role": "user", "content": "Who owns tax?"}],
    }
    teacher = ScriptedTeacher([
        {"content": "Pricing owns tax because Orders must stay free of tax rules.", "reasoning_content": "flow"},
        {"content": "Pricing", "reasoning_content": "Orders would mix tax into persistence."},
    ])
    rows = list(generate_traces([task, task], teacher))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "Pricing"


def test_prompt_lists_every_question_already_used():
    from sftmill.generate.synthesize_tasks import _build_task_messages

    category = {"id": "math", "kind": "trace", "match": "exact", "template": "math"}
    previous = [f"question {n}" for n in range(6)]
    messages = _build_task_messages(category, previous)
    content = messages[-1]["content"]
    for question in previous:
        assert question in content


def test_system_prompt_requires_the_question_and_the_gold_to_match():
    from sftmill.generate.synthesize_tasks import _build_task_messages

    incident = {
        "id": "incident_fix",
        "kind": "trajectory",
        "match": "exact",
        "template": "report",
        "verify": "python3 check.py",
    }
    system = _build_task_messages(incident, [])[0]["content"]
    assert "same bug" in system
    assert "log line" in system
    assert "python3 check.py" in system
    owner = {"id": "design_owner", "kind": "trace", "match": "exact", "template": "owner"}
    owner_system = _build_task_messages(owner, [])[0]["content"]
    assert "Exactly one named component" in owner_system
    assert "same bug" not in owner_system
    repair = {
        "id": "python_repair",
        "kind": "trajectory",
        "match": "exact",
        "template": "calc",
        "verify": "python3 calc.py",
    }
    repair_system = _build_task_messages(repair, [])[0]["content"]
    assert "Do not retune a rate" in repair_system
    feature = {
        "id": "long_feature",
        "kind": "trajectory",
        "match": "exact",
        "template": "later",
        "verify": "python3 check.py",
    }
    feature_system = _build_task_messages(feature, [])[0]["content"]
    assert "The wrong value is neither the seed nor the gold" in feature_system
    compat = {
        "id": "compat_change",
        "kind": "trajectory",
        "match": "exact",
        "template": "both",
        "verify": "python3 check.py",
    }
    compat_system = _build_task_messages(compat, [])[0]["content"]
    assert "not a rule" in compat_system


def test_expect_files_copy_is_rejected():
    category = {"id": "search_edit", "toolset": "workspace", "require_observation": "PASSED"}
    item = {
        "question": "Run python3 check.py.",
        "answer": "ok",
        "files": {"check.py": "print('PASSED')\n", "a.py": "x\n", "b.py": "y\n"},
        "expect_files": {"check.py": "print('PASSED')\n", "a.py": "x\n", "b.py": "y\n"},
    }
    assert _reject_reason(category, item) == "expect_files matches files"


def test_json_in_reasoning_is_used_when_content_is_empty():
    category = {
        "id": "math_arithmetic",
        "count": 1,
        "kind": "trace",
        "match": "exact",
        "template": "math",
    }
    teacher = ScriptedTeacher([
        {"content": "", "reasoning_content": '{"question": "2+2?", "answer": "4"}'},
    ])
    rows, err = synthesize_category_tasks(category, teacher, max_attempts=1)
    assert err is None
    assert rows[0]["answer"] == "4"


def test_extract_and_normalize_batch():
    text = 'Here you go:\n[{"question": "What is 2+2?", "answer": "4"}]\n'
    batch = _extract_json_array(text)
    assert batch is not None
    item = _normalize_item(batch[0])
    assert item == {"question": "What is 2+2?", "answer": "4"}


def test_task_from_item_trace_and_trajectory():
    trace_cat = {"id": "math", "kind": "trace", "match": "exact", "template": ""}
    traj_cat = {"id": "py", "kind": "trajectory", "match": "exact", "template": ""}
    trace = _task_from_item(trace_cat, {"question": "6*7?", "answer": "42"}, 1)
    traj = _task_from_item(traj_cat, {"question": "Sum 1..10", "answer": "55"}, 2)
    validate_example(trace)
    validate_example(traj)
    assert trace["kind"] == "trace"
    assert "tools" not in trace
    assert traj["tools"][0]["function"]["name"] == "python"
    assert "python tool" in traj["messages"][0]["content"].lower()


class ScriptedTeacher:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.last_kwargs: dict = {}

    def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if not self._replies:
            return {"content": "{}"}
        reply = self._replies.pop(0)
        return reply if isinstance(reply, dict) else {"content": reply}


def test_bug_needle_repair_on_second_reply():
    category = {
        "id": "bug_needle",
        "count": 1,
        "kind": "trace",
        "match": "contains",
        "template": "needle",
    }
    teacher = ScriptedTeacher([
        {"content": json.dumps({"question": "fault is KeyError", "answer": "KeyError"})},
        {"content": json.dumps({"question": "Trace this snippet.\ndef f():\n    pass", "answer": "KeyError"})},
    ])
    rows, err = synthesize_category_tasks(category, teacher, max_attempts=2)
    assert err is None
    assert rows[0]["answer"] == "KeyError"
    assert teacher.calls == 2


def test_synthesis_request_carries_schema_and_high_token_budget():
    category = {
        "id": "bug_needle",
        "count": 1,
        "kind": "trace",
        "match": "contains",
        "template": "needle",
    }
    teacher = ScriptedTeacher([
        {"content": json.dumps({"question": "Trace this.\ndef f():\n    pass", "answer": "KeyError"})},
    ])
    synthesize_category_tasks(category, teacher, max_attempts=1)
    assert teacher.last_kwargs["max_tokens"] == SYNTHESIS_MAX_TOKENS
    assert teacher.last_kwargs["response_format"]["type"] == "json_schema"
    assert "enable_thinking" not in teacher.last_kwargs
    _, payload, _ = build_chat_request(
        "http://127.0.0.1:8080/v1",
        "m",
        [{"role": "user", "content": "hi"}],
        response_format=category_response_schema(category),
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )
    assert payload["max_tokens"] == SYNTHESIS_MAX_TOKENS
    assert payload["response_format"]["json_schema"]["name"] == "bug_needle"


def test_synthesize_category_skips_malformed_then_succeeds():
    category = {
        "id": "math_arithmetic",
        "count": 2,
        "kind": "trace",
        "match": "exact",
        "template": "math",
    }
    teacher = ScriptedTeacher([
        {"content": "not json"},
        {"content": json.dumps({"question": "What is 3+4?", "answer": "7"})},
        {"content": json.dumps({"question": "What is 5+5?", "answer": "10"})},
    ])
    rows, err = synthesize_category_tasks(category, teacher, max_attempts=6)
    assert err is None
    assert len(rows) == 2
    assert {row["answer"] for row in rows} == {"7", "10"}


def test_synthesize_tasks_runs_three_categories_at_once(tmp_path):
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text(
        """
name: parallel
categories:
  - id: one
    count: 1
    kind: trace
    match: exact
    template: a
  - id: two
    count: 1
    kind: trace
    match: exact
    template: b
  - id: three
    count: 1
    kind: trace
    match: exact
    template: c
""",
    )

    class Teacher:
        def complete(self, messages, tools=None, **kwargs):
            category = messages[-1]["content"].split("Category: ", 1)[1].split()[0]
            return {"content": json.dumps({"question": f"q-{category}", "answer": "1"})}

    rows, shortfalls = synthesize_tasks(curriculum, Teacher(), tmp_path / "tasks.jsonl", batch_size=1, jobs=3)
    assert not shortfalls
    assert {row["id"] for row in rows} == {"one-0001", "two-0001", "three-0001"}


def test_synthesize_tasks_fans_out_across_teacher_list(tmp_path):
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text(
        """
name: parallel
categories:
  - id: one
    count: 1
    kind: trace
    match: exact
    template: a
  - id: two
    count: 1
    kind: trace
    match: exact
    template: b
  - id: three
    count: 1
    kind: trace
    match: exact
    template: c
""",
    )
    used: list[str] = []
    lock = threading.Lock()

    class NamedTeacher:
        def __init__(self, name):
            self.name = name

        def complete(self, messages, tools=None, **kwargs):
            with lock:
                used.append(self.name)
            category = messages[-1]["content"].split("Category: ", 1)[1].split()[0]
            return {"content": json.dumps({"question": f"q-{category}-{self.name}", "answer": "1"})}

    rows, shortfalls = synthesize_tasks(
        curriculum,
        [NamedTeacher("a"), NamedTeacher("b")],
        tmp_path / "tasks.jsonl",
    )
    assert not shortfalls
    assert set(used) == {"a", "b"}
    assert {row["id"] for row in rows} == {"one-0001", "two-0001", "three-0001"}


def test_synthesize_tasks_dedupes_and_writes(tmp_path):
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text(
        """
name: tiny
categories:
  - id: math_arithmetic
    count: 1
    kind: trace
    match: exact
    template: one math task
""",
    )
    teacher = ScriptedTeacher([
        {"content": json.dumps({"question": "2+2?", "answer": "4"})},
    ])
    out = tmp_path / "tasks.jsonl"
    rows, shortfalls = synthesize_tasks(curriculum, teacher, out, batch_size=1)
    assert not shortfalls
    assert len(rows) == 1
    assert json.loads(out.read_text().strip())["answer"] == "4"
    assert rows[0]["id"] == "math_arithmetic-0001"


def test_synthesized_tasks_work_with_generate_pipeline():
    trace_task = _task_from_item(
        {"id": "math", "kind": "trace", "match": "exact", "template": ""},
        {"question": "6*7", "answer": "42"},
        1,
    )
    traj_task = _task_from_item(
        {"id": "py", "kind": "trajectory", "match": "exact", "template": ""},
        {"question": "add", "answer": "4"},
        1,
    )
    trace_teacher = ScriptedTeacher(['{"reasoning": "r", "answer": "42"}'])
    traj_teacher = ScriptedTeacher([
        {"content": "", "tool_calls": [{"name": "python", "arguments": {"code": "print(2+2)"}}]},
        '{"reasoning": "done", "answer": "4"}',
    ])

    traces = list(generate_traces([trace_task], trace_teacher))
    trajectories = list(generate_trajectories([traj_task], traj_teacher, max_steps=3))
    assert len(traces) == 1
    assert len(trajectories) == 1


def test_workspace_items_without_files_are_skipped():
    category = {
        "id": "search_edit",
        "count": 1,
        "kind": "trajectory",
        "match": "exact",
        "toolset": "workspace",
        "require_observation": "PASSED",
        "template": "files",
    }
    good = {
        "question": "The check fails. Run python3 check.py.",
        "answer": "ok",
        "files": {
            "check.py": "print('PASSED')\n",
            "app.py": "VALUE = 1\n",
            "lib.py": "pass\n",
        },
        "expect_files": {"app.py": "VALUE = 2\n"},
    }
    teacher = ScriptedTeacher([
        {"content": json.dumps({"question": "pasted tree\n" + "x" * 50, "answer": "ok"})},
        {"content": json.dumps(good)},
    ])
    rows, err = synthesize_category_tasks(category, teacher, max_attempts=4)
    assert err is None
    assert len(rows) == 1
    assert rows[0]["files"]["check.py"].startswith("print")


def test_synthesize_category_returns_partial_without_raising():
    category = {
        "id": "math_arithmetic",
        "count": 3,
        "kind": "trace",
        "match": "exact",
        "template": "math",
    }
    teacher = ScriptedTeacher([
        {"content": json.dumps({"question": "only one", "answer": "1"})},
    ])
    rows, err = synthesize_category_tasks(category, teacher, max_attempts=1)
    assert len(rows) == 1
    assert err is not None
    assert "got 1/3" in err


def test_synthesize_tasks_keeps_rows_when_one_category_is_short(tmp_path):
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text(
        """
name: partial
categories:
  - id: ok
    count: 1
    kind: trace
    match: exact
    template: a
  - id: bad
    count: 2
    kind: trace
    match: exact
    template: b
""",
    )

    class Teacher:
        def complete(self, messages, tools=None, **kwargs):
            category = "ok"
            for message in messages:
                content = message.get("content") or ""
                if "Category: " in content:
                    category = content.split("Category: ", 1)[1].split()[0]
                    break
            if category == "bad":
                return {"content": json.dumps({"question": "x", "answer": ""})}
            return {"content": json.dumps({"question": "good", "answer": "1"})}

    out = tmp_path / "tasks.jsonl"
    rows, shortfalls = synthesize_tasks(curriculum, Teacher(), out, jobs=1)
    assert len(rows) == 1
    assert rows[0]["id"] == "ok-0001"
    assert len(shortfalls) == 1
    assert "bad" in shortfalls[0]
    assert len(out.read_text().strip().splitlines()) == 1


def test_load_curriculum_code_instruct():
    data = load_curriculum("configs/curriculum/code_instruct.yaml")
    assert data["name"] == "code_instruct"
    assert len(data["categories"]) == 22
    multi = 0
    for category in data["categories"]:
        assert category["count"] == 80
        assert category["kind"] == "trace"
        assert category["match"] == "open"
        assert "harnesses" not in category
        example = category["example"]
        item = _normalize_item(example)
        assert _reject_reason(category, item) is None
        task = _task_from_item(category, item, 1)
        assert "answer" not in task
        assert "Your final reply must be only" not in json.dumps(task["messages"])
        if category.get("user_turns"):
            multi += 1
            assert list(example) == ["turns"]
            assert len(example["turns"]) == category["user_turns"]
            assert all(isinstance(turn, str) and turn.strip() for turn in example["turns"])
            assert [message["role"] for message in task["messages"]] == ["user"] * category["user_turns"]
        else:
            assert "turns" not in example
            assert example["question"].strip()
    assert multi == 16


def test_open_rejects_a_gold_answer_and_the_wrong_turn_count():
    single = {"id": "agent_help", "match": "open", "kind": "trace"}
    assert _reject_reason(single, {"question": "Hi", "answer": "hello"}) == "open tasks have no gold answer"
    dialogue = {"id": "chat", "match": "open", "kind": "trace", "user_turns": 3}
    assert _reject_reason(dialogue, {"question": "a\nb", "turns": ["a", "b"]}) == "turns must be 3 user messages"
    schema = category_response_schema(dialogue)
    assert schema["json_schema"]["schema"]["required"] == ["turns"]
    assert "answer" not in schema["json_schema"]["schema"]["properties"]
