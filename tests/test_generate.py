import json

from sftmill.generate.interpret import align_answer, extract_graded_answer, interpret_teacher
from sftmill.generate.traces import generate_traces
from sftmill.generate.trajectories import generate_trajectory
from sftmill.schema import load_jsonl
from sftmill.teachers.openai_compat import (
    build_chat_request,
    message_from_completion,
    message_from_stream_chunks,
    messages_for_chat_api,
)


class ScriptedTeacher:
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages, tools=None):
        return self._replies.pop(0)


_PROSE = (
    "I have not edited any files. "
    "The next step is to read the traceback and name the missing key."
)
_PROSE_NEXT = (
    "Use a default when the key is missing. "
    "Then the call returns instead of raising."
)


def test_open_trace_keeps_a_multi_paragraph_reply():
    reply = _PROSE + "\n\n" + _PROSE_NEXT
    task = {
        "id": "open",
        "kind": "trace",
        "match": "open",
        "messages": [{"role": "user", "content": "What does this traceback mean?"}],
    }
    rows = list(generate_traces([task], ScriptedTeacher([
        {"content": reply, "reasoning_content": "the last line is the symptom"},
    ])))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == reply
    assert rows[0]["messages"][-1]["reasoning_content"] == "the last line is the symptom"
    assert "answer" not in rows[0]


def test_open_trace_drops_a_one_token_reply():
    task = {
        "id": "open",
        "kind": "trace",
        "match": "open",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert list(generate_traces([task], ScriptedTeacher(["Ok."]))) == []


def test_open_two_turn_trace_keeps_both_teacher_replies():
    class Recording:
        def __init__(self):
            self.seen = []
            self._replies = [
                {"content": _PROSE, "reasoning_content": "ask what is missing"},
                {"content": _PROSE_NEXT, "reasoning_content": "they named the timeout"},
            ]

        def complete(self, messages, tools=None):
            self.seen.append([(message["role"], message.get("content")) for message in messages])
            return self._replies.pop(0)

    teacher = Recording()
    task = {
        "id": "two",
        "kind": "trace",
        "match": "open",
        "turns": ["Add a timeout.", "Ten seconds, and do not edit yet."],
        "messages": [{"role": "user", "content": "ignored"}],
    }
    rows = list(generate_traces([task], teacher))
    assert [message["role"] for message in rows[0]["messages"]] == [
        "user", "assistant", "user", "assistant",
    ]
    assert rows[0]["messages"][1]["content"] == _PROSE
    assert rows[0]["messages"][3]["content"] == _PROSE_NEXT
    assert rows[0]["messages"][1]["reasoning_content"] == "ask what is missing"
    assert teacher.seen[0] == [("user", "Add a timeout.")]
    assert teacher.seen[1][1] == ("assistant", _PROSE)
    assert teacher.seen[1][2] == ("user", "Ten seconds, and do not edit yet.")

    short = Recording()
    short._replies[1] = {"content": "Ok."}
    dropped = {
        "id": "short",
        "kind": "trace",
        "match": "open",
        "turns": ["Add a timeout.", "Ten seconds."],
        "messages": [],
    }
    assert list(generate_traces([dropped], short)) == []


def test_trace_keeps_exact_answer_and_drops_a_miss():
    tasks = [
        {"id": "ok", "kind": "trace", "answer": "42", "messages": [{"role": "user", "content": "6*7"}]},
        {"id": "bad", "kind": "trace", "answer": "42", "messages": [{"role": "user", "content": "6*7"}]},
    ]
    teacher = ScriptedTeacher([
        '{"reasoning": "six times seven", "answer": "42"}',
        '{"reasoning": "nope", "answer": "41"}',
    ])
    rows = list(generate_traces(tasks, teacher))
    assert [row["task_id"] for row in rows] == ["ok"]
    assert rows[0]["messages"][-1]["reasoning_content"] == "six times seven"
    assert rows[0]["messages"][-1]["content"] == "42"


def test_trajectory_runs_the_sandbox_and_masks_nothing_in_the_row_itself():
    task = {
        "id": "py",
        "kind": "trajectory",
        "answer": "4",
        "messages": [{"role": "user", "content": "add"}],
    }

    def teacher_complete(messages, tools=None):
        if not any(message["role"] == "tool" for message in messages):
            return {
                "content": "",
                "tool_calls": [{"name": "python", "arguments": {"code": "print(2 + 2)"}}],
            }
        return '{"reasoning": "printed", "answer": "4"}'

    class Teacher:
        def complete(self, messages, tools=None):
            return teacher_complete(messages, tools)

    row = generate_trajectory(task, Teacher(), max_steps=3)
    assert row is not None
    roles = [message["role"] for message in row["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert row["messages"][2]["content"].strip() == "4"
    assert row["messages"][1]["tool_calls"][0]["name"] == "python"


def test_align_answer_verbose_numeric_sentence():
    sentence = "The farmer harvested a total of **465 kilograms** of wheat from all three fields."
    assert align_answer(sentence, "465") == "465"


def test_trace_drops_verbose_numeric_content():
    tasks = [{
        "id": "wheat",
        "kind": "trace",
        "match": "exact",
        "answer": "465",
        "messages": [{"role": "user", "content": "sum fields"}],
    }]
    teacher = ScriptedTeacher([{
        "content": "The farmer harvested a total of **465 kilograms** of wheat from all three fields.",
        "reasoning_content": "140 + 105 + 220 = 465",
    }])
    assert list(generate_traces(tasks, teacher)) == []


def test_interpret_reasoning_channel_only():
    assistant = interpret_teacher({
        "content": "",
        "reasoning_content": "Work through steps...\nFinal answer: 42",
    })
    assert assistant["content"] == "42"
    assert "Work through" in assistant["reasoning_content"]


def test_extract_graded_answer_finds_needle_in_reasoning():
    got = extract_graded_answer({
        "content": "Long prose that never names the fault class.",
        "reasoning_content": "This raises a KeyError when the key is missing.",
    }, "KeyError")
    assert got == "KeyError"


def test_contains_trace_stores_needle_in_content():
    tasks = [{
        "id": "needle",
        "kind": "trace",
        "match": "contains",
        "answer": "itmes",
        "messages": [{"role": "user", "content": "trace snippet"}],
    }]
    teacher = ScriptedTeacher([{
        "content": "A typo in the identifier causes a NameError every call.",
        "reasoning_content": "The broken name is itmes instead of items.",
    }])
    rows = list(generate_traces(tasks, teacher))
    assert rows[0]["messages"][-1]["content"] == "itmes"


def test_extract_graded_answer_json_in_reasoning():
    got = extract_graded_answer({
        "content": "",
        "reasoning_content": 'Thoughts first.\n{"reasoning": "x", "answer": "99"}',
    })
    assert got == "99"


def test_trace_keeps_llama_style_reasoning_message():
    tasks = [{"id": "r", "kind": "trace", "answer": "42", "messages": [{"role": "user", "content": "6*7"}]}]
    teacher = ScriptedTeacher([{
        "content": "",
        "reasoning_content": "6 times 7 equals 42.\nFinal answer: 42",
    }])
    rows = list(generate_traces(tasks, teacher))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "42"


def test_interpret_openai_tool_call_json_string():
    message = interpret_teacher({
        "content": None,
        "tool_calls": [{
            "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})},
        }],
    })
    assert message["tool_calls"][0]["arguments"]["code"] == "print(1)"


def test_stream_chunks_merge_content_and_tools():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"content": '{"answer":'}}]},
        {"choices": [{"delta": {"content": ' "4"}'}}]},
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "python", "arguments": '{"code":'},
                    }],
                },
            }],
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": ' "print(1)"}'},
                    }],
                },
            }],
        },
    ]
    message = message_from_stream_chunks(chunks)
    assert message["reasoning_content"] == "think "
    assert message["content"] == '{"answer": "4"}'
    assert message["tool_calls"][0]["function"]["name"] == "python"
    assert "print(1)" in message["tool_calls"][0]["function"]["arguments"]


def test_openai_request_shape():
    url, payload, headers = build_chat_request(
        "http://127.0.0.1:8000/v1", "teacher", [{"role": "user", "content": "hi"}], api_key="k",
    )
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert payload["model"] == "teacher"
    assert headers["Authorization"] == "Bearer k"
    parsed = message_from_completion({"choices": [{"message": {"content": "ok"}}]})
    assert parsed["content"] == "ok"


def test_messages_for_chat_api_tool_calls():
    api = messages_for_chat_api([
        {"role": "user", "content": "run code"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "python", "arguments": {"code": "print(1)"}}],
        },
        {"role": "tool", "name": "python", "content": "1\n"},
    ])
    call = api[1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "python"
    assert json.loads(call["function"]["arguments"]) == {"code": "print(1)"}
    assert api[2]["tool_call_id"] == call["id"]


def test_extract_graded_answer_boxed_before_check_line():
    content = """Area = 21 × 8 = \\boxed{168} cm^2

**Check:** Perimeter = 2(21 + 8) = 2(29) = 58 cm ✓"""
    assert extract_graded_answer({"content": content}, "168") == "168"


def test_example_tasks_parse():
    rows = load_jsonl("configs/tasks/examples.jsonl")
    assert {row["kind"] for row in rows} == {"trace", "trajectory"}


def _python_until(answer_after: int, answer: str):
    def complete(messages, tools=None):
        seen = sum(1 for message in messages if message["role"] == "tool")
        if seen < answer_after:
            return {"content": "", "tool_calls": [{"name": "python", "arguments": {"code": f"print({seen})"}}]}
        return json.dumps({"answer": answer})

    class Teacher:
        def complete(self, messages, tools=None):
            return complete(messages, tools)

    return Teacher()


def test_task_max_steps_allows_six_tool_rounds_then_an_answer():
    task = {
        "id": "long",
        "kind": "trajectory",
        "answer": "done",
        "messages": [{"role": "user", "content": "go"}],
        "max_steps": 6,
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}],
    }
    row = generate_trajectory(task, _python_until(6, "done"))
    assert row is not None
    assert sum(1 for message in row["messages"] if message["role"] == "tool") == 6
    assert row["messages"][-1]["content"] == "done"


def test_default_max_steps_stops_before_the_answer():
    task = {
        "id": "short",
        "kind": "trajectory",
        "answer": "done",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}],
    }
    assert generate_trajectory(task, _python_until(6, "done"), max_steps=4) is None


def test_harness_script_grades_the_result_value():
    from sftmill.generate.filters import accept, short_answer

    answer = (
        'Trajectory: (1) Bad call. (2) Corrected call. Result: 18.4°C'
    )
    assert short_answer(answer) == "18.4°C"
    example = {
        "messages": [
            {"role": "user", "content": "temp?"},
            {"role": "assistant", "content": "18.4°C"},
        ],
    }
    assert accept(example, {"answer": answer, "match": "exact", "id": "h"})


def test_contains_keeps_a_needle_in_reasoning():
    from sftmill.generate.filters import accept

    example = {
        "messages": [{
            "role": "assistant",
            "content": "Let me write a clear response identifying the fault.",
            "reasoning_content": "The call returns NullPointerException.",
        }],
    }
    assert accept(example, {"answer": "NullPointerException", "match": "contains", "id": "b"})
    assert not accept(example, {"answer": "NameError", "match": "contains", "id": "b"})


def test_contains_keeps_the_needle_and_drops_the_wrong_one():
    kept = {
        "id": "bug",
        "kind": "trace",
        "match": "contains",
        "answer": "ZeroDivisionError",
        "messages": [{"role": "user", "content": "what broke?"}],
    }
    dropped = dict(kept)
    dropped["answer"] = "NameError"
    teacher = ScriptedTeacher([
        {"content": "The fault is ZeroDivisionError."},
        {"content": "The fault is ZeroDivisionError."},
    ])
    rows = list(generate_traces([kept, dropped], teacher))
    assert [row["task_id"] for row in rows] == ["bug"]


def test_unknown_tool_and_missing_arguments_do_not_write():
    from sftmill.schema import WORKSPACE_TOOLS

    task = {
        "id": "bad-call",
        "kind": "trajectory",
        "answer": "ok",
        "messages": [{"role": "user", "content": "edit"}],
        "tools": WORKSPACE_TOOLS,
        "files": {"a.py": "value = 1\n"},
        "expect_files": {"a.py": "value = 2\n"},
        "max_steps": 4,
    }

    def complete(messages, tools=None):
        calls = [message for message in messages if message["role"] == "tool"]
        if not calls:
            return {"content": "", "tool_calls": [{"name": "nope", "arguments": {"path": "a.py"}}]}
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "a.py"}}]}
        if len(calls) == 2:
            return {
                "content": "",
                "tool_calls": [{
                    "name": "edit_file",
                    "arguments": {"path": "a.py", "old": "value = 1\n", "new": "value = 2\n"},
                }],
            }
        return json.dumps({"answer": "ok"})

    class Teacher:
        def complete(self, messages, tools=None):
            return complete(messages, tools)

    row = generate_trajectory(task, Teacher())
    assert row is not None
    assert row["messages"][2]["content"].startswith("error: unknown tool")
    assert "invalid arguments" in row["messages"][4]["content"]
    assert row["messages"][-1]["content"] == "ok"


def test_verify_keeps_an_equivalent_patch_and_drops_a_faked_pass():
    from sftmill.schema import WORKSPACE_TOOLS

    task = {
        "id": "edit",
        "kind": "trajectory",
        "match": "exact",
        "answer": "ok",
        "require_observation": "PASSED",
        "verify": "python3 check.py",
        "messages": [{"role": "user", "content": "edit"}],
        "tools": WORKSPACE_TOOLS,
        "files": {
            "app.py": "VALUE = 1\n",
            "check.py": "import app\nassert app.VALUE == 2\nprint('PASSED')\n",
        },
        "expect_files": {"app.py": "VALUE = 2\n"},
        "max_steps": 4,
    }

    def teacher_for(command: str, final: str):
        def complete(messages, tools=None):
            if not any(message["role"] == "tool" for message in messages):
                return {"content": "", "tool_calls": [{"name": "bash", "arguments": {"command": command}}]}
            return {"content": final}

        class Teacher:
            def complete(self, messages, tools=None):
                return complete(messages, tools)

        return Teacher()

    write_and_check = "python3 -c \"open('app.py','w').write('VALUE = 2\\n')\" && python3 check.py"
    edited = generate_trajectory(task, teacher_for(write_and_check, "ok"))
    assert edited is not None
    assert edited["messages"][-1]["content"] == "ok"
    prose = generate_trajectory(task, teacher_for(write_and_check, "Fixed it."))
    assert prose is None
    faked = generate_trajectory(task, teacher_for("echo PASSED", "ok"))
    assert faked is None


def test_lookup_verify_ignores_files_the_solver_added():
    from sftmill.schema import WORKSPACE_TOOLS

    task = {
        "id": "cli_find-0001",
        "kind": "trajectory",
        "match": "exact",
        "answer": "2",
        "verify": "find . -type f | wc -l",
        "verify_stdout": "exact",
        "messages": [{"role": "user", "content": "How many files are seeded?"}],
        "tools": WORKSPACE_TOOLS,
        "files": {"a.txt": "one\n", "b.txt": "two\n"},
        "max_steps": 4,
    }

    def complete(messages, tools=None):
        if not any(message["role"] == "tool" for message in messages):
            return {"content": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "extra.txt", "content": "nope\n"}}]}
        return {"content": "2"}

    class Teacher:
        def complete(self, messages, tools=None):
            return complete(messages, tools)

    row = generate_trajectory(task, Teacher())
    assert row is not None
    assert row["messages"][-1]["content"] == "2"


def test_two_calls_in_one_turn_stay_in_order_and_a_final_tool_call_is_dropped():
    task = {
        "id": "pair",
        "kind": "trajectory",
        "answer": "ok",
        "messages": [{"role": "user", "content": "list"}],
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}],
        "max_steps": 2,
    }

    def complete(messages, tools=None):
        if not any(message["role"] == "tool" for message in messages):
            return {
                "content": "",
                "tool_calls": [
                    {"name": "python", "arguments": {"code": "print('one')"}},
                    {"name": "python", "arguments": {"code": "print('two')"}},
                ],
            }
        return json.dumps({"answer": "ok"})

    class Teacher:
        def complete(self, messages, tools=None):
            return complete(messages, tools)

    row = generate_trajectory(task, Teacher())
    assert row is not None
    assert [message["content"].strip() for message in row["messages"] if message["role"] == "tool"] == ["one", "two"]

    def always_call(messages, tools=None):
        return {"content": "", "tool_calls": [{"name": "python", "arguments": {"code": "print(1)"}}]}

    class Loop:
        def complete(self, messages, tools=None):
            return always_call(messages, tools)

    assert generate_trajectory(task, Loop(), max_steps=1) is None
