from sftmill.generate.synthesize_tasks import (
    _normalize_item,
    _reject_reason,
    _task_from_item,
    category_response_schema,
    load_curriculum,
)
from sftmill.generate.traces import generate_traces
from sftmill.identity import default_system, prepend_default_system, teacher_system


def test_identity_prompt_is_short():
    text = default_system()
    assert "Alice" in text
    assert "language model" in text
    assert "listed" in text
    assert "[COT_START]" not in text
    assert "[COT_END]" not in text
    assert "[COT_START]" not in teacher_system(text)
    assert "[COT_END]" not in teacher_system(text)
    assert 25 <= len(text.split()) <= 50


def test_prepend_default_system_skips_existing_system():
    custom = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
    assert prepend_default_system(custom) == custom
    injected = prepend_default_system([{"role": "user", "content": "hi"}])
    assert injected[0] == {"role": "system", "content": default_system()}
    assert injected[1]["content"] == "hi"


def test_general_instruct_curriculum_is_wide_and_thin():
    data = load_curriculum("configs/curriculum/general_instruct.yaml")
    categories = data["categories"]
    assert data["name"] == "general_instruct"
    assert len(categories) == 80
    total = sum(int(category["count"]) for category in categories)
    assert total == 4000
    custom = {category["id"] for category in categories if category.get("student_system") == "custom"}
    assert custom == {"system_follow", "system_roleplay", "system_constraint", "system_vs_user"}
    for category in categories:
        assert category["kind"] == "trace"
        assert category["match"] == "open"
        assert "harnesses" not in category
        example = category["example"]
        item = _normalize_item(example)
        assert _reject_reason(category, item) is None
        task = _task_from_item(category, item, 1)
        assert task["messages"][0]["role"] == "system"
        assert "answer" not in task
        if category.get("student_system") == "custom":
            assert task["system"] == example["system"]
            assert task["student_system"] == "custom"
        else:
            assert task["system"] == default_system()


def test_custom_system_schema_and_reject():
    category = {
        "id": "system_follow",
        "kind": "trace",
        "match": "open",
        "student_system": "custom",
    }
    schema = category_response_schema(category)
    assert schema["json_schema"]["schema"]["required"] == ["question", "system"]
    assert _reject_reason(category, {"question": "hi"}) == "custom student_system needs a system string"
    item = _normalize_item({"question": "Pack a bag.", "system": "Three bullets."})
    assert item["system"] == "Three bullets."
    task = _task_from_item(category, item, 1)
    assert task["messages"][0]["content"] == "Three bullets."


class ScriptedTeacher:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen = []

    def complete(self, messages, tools=None):
        self.seen.append(list(messages))
        return self._replies.pop(0)


def test_open_trace_injects_identity_and_teacher_system():
    teacher = ScriptedTeacher([{
        "content": "I am Alice, a language model. I do not have a body.",
        "reasoning_content": "identity",
    }])
    rows = list(generate_traces([{
        "id": "identity-0001",
        "kind": "trace",
        "match": "open",
        "messages": [{"role": "user", "content": "Who are you?"}],
    }], teacher))
    assert rows[0]["messages"][0]["content"] == default_system()
    assert teacher.seen[0][0]["content"] == teacher_system(default_system())
    assert "[COT_START]" not in teacher.seen[0][0]["content"]
    assert "[COT_END]" not in teacher.seen[0][0]["content"]


def test_open_trace_uses_live_identity_not_baked_system():
    teacher = ScriptedTeacher([{
        "content": "I am Alice, a language model. I do not have a body.",
        "reasoning_content": "identity",
    }])
    stale = "Think in [COT_START]...[COT_END], then answer the user."
    rows = list(generate_traces([{
        "id": "identity-0001",
        "kind": "trace",
        "match": "open",
        "student_system": "default",
        "system": stale,
        "messages": [
            {"role": "system", "content": stale},
            {"role": "user", "content": "Who are you?"}],
    }], teacher))
    assert rows[0]["messages"][0]["content"] == default_system()
    assert stale not in rows[0]["messages"][0]["content"]
    assert "[COT_START]" not in teacher.seen[0][0]["content"]
    assert "[COT_END]" not in teacher.seen[0][0]["content"]


def test_open_trace_keeps_custom_system():
    teacher = ScriptedTeacher([{
        "content": "Pack a shell. Pack a stove. Pack a map.",
        "reasoning_content": "three bullets",
    }])
    rows = list(generate_traces([{
        "id": "system_follow-0001",
        "kind": "trace",
        "match": "open",
        "system": "Answer in exactly three bullet points. No intro.",
        "messages": [
            {"role": "system", "content": "Answer in exactly three bullet points. No intro."},
            {"role": "user", "content": "What should I pack?"},
        ],
    }], teacher))
    assert rows[0]["messages"][0]["content"].startswith("Answer in exactly three")
    assert default_system() not in rows[0]["messages"][0]["content"]


def test_open_trace_can_disable_student_system():
    teacher = ScriptedTeacher([{
        "content": "Two sentences about timeouts. Then a next step.",
        "reasoning_content": "done",
    }])
    rows = list(generate_traces([{
        "id": "plain-0001",
        "kind": "trace",
        "match": "open",
        "student_system": False,
        "messages": [{"role": "user", "content": "hi"}],
    }], teacher))
    assert rows[0]["messages"][0]["role"] == "user"
    assert teacher.seen[0][0]["role"] == "system"
    assert teacher.seen[0][0]["content"] == teacher_system(None)
