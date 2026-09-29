import json

from sftmill.generate.trajectories import generate_trajectory
from sftmill.schema import WORKSPACE_TOOLS
from sftmill.tools.workspace import TRUNCATION_MARKER, Workspace


def test_write_read_and_list_round_trip(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.write_file("notes/a.txt", "hello") == "ok"
    assert workspace.read_file("notes/a.txt") == "hello"
    assert "a.txt" in workspace.list_dir("notes").splitlines()


def test_edit_file_replaces_one_occurrence_only(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write_file("a.txt", "one one")
    original = workspace.read_text("a.txt")
    assert "missing" in workspace.edit_file("a.txt", "absent", "x")
    assert "more than once" in workspace.edit_file("a.txt", "one", "two")
    assert workspace.read_text("a.txt") == original
    workspace.write_file("a.txt", "one")
    assert workspace.edit_file("a.txt", "one", "two") == "ok"
    assert workspace.read_text("a.txt") == "two"


def test_paths_outside_the_root_do_not_run(tmp_path):
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / "escaped.txt"
    assert "escapes" in workspace.write_file("../escaped.txt", "nope")
    assert "escapes" in workspace.write_file(str(outside), "nope")
    assert not outside.exists()
    assert "escapes" in workspace.search("x", "../")
    assert "escapes" in workspace.bash("touch ../escaped.txt")
    assert not outside.exists()


def test_missing_command_returns_an_error(tmp_path):
    workspace = Workspace(tmp_path)
    result = workspace.bash("not-a-real-binary")
    assert result.startswith("error:")
    assert "not-a-real-binary" in result


def test_bash_env_assignment_uses_shell(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write_file("check.py", "import os\nassert os.environ.get('MARK') == '1'\nprint('PASSED')\n")
    result = workspace.bash("MARK=1 python3 check.py")
    assert "PASSED" in result, result


def test_normalize_file_text_trailing_newline():
    from sftmill.generate.trajectories import _file_texts_equal

    assert _file_texts_equal("a\n", "a")
    assert _file_texts_equal("a", "a\n")


def test_search_caps_hits_and_read_file_windows(tmp_path):
    workspace = Workspace(tmp_path, max_output=500, max_hits=1)
    workspace.write_file("a.txt", "alpha\nbeta\ngamma\n")
    workspace.write_file("b.txt", "alpha\n")
    hits = workspace.search("alpha").splitlines()
    assert len(hits) == 1
    assert hits[0].startswith("a.txt:1:")
    assert workspace.read_file("a.txt", offset=2, limit=1) == "beta"
    short = Workspace(tmp_path, max_output=12)
    long = short.read_file("a.txt")
    assert long.endswith(TRUNCATION_MARKER)


def test_expect_files_rejects_a_stale_file():
    task = {
        "id": "edit",
        "kind": "trajectory",
        "answer": "ok",
        "messages": [{"role": "user", "content": "edit"}],
        "tools": WORKSPACE_TOOLS,
        "files": {"a.py": "old\n"},
        "expect_files": {"a.py": "new\n"},
        "max_steps": 4,
    }

    def teacher_for(edited: bool):
        def complete(messages, tools=None):
            if not any(message["role"] == "tool" for message in messages):
                if edited:
                    return {
                        "content": "",
                        "tool_calls": [{
                            "name": "edit_file",
                            "arguments": {"path": "a.py", "old": "old\n", "new": "new\n"},
                        }],
                    }
                return {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}
            return json.dumps({"answer": "ok"})

        class Teacher:
            def complete(self, messages, tools=None):
                return complete(messages, tools)

        return Teacher()

    assert generate_trajectory(task, teacher_for(True)) is not None
    assert generate_trajectory(task, teacher_for(False)) is None


def test_require_observation_needs_the_check_line():
    task = {
        "id": "check",
        "kind": "trajectory",
        "answer": "ok",
        "require_observation": "3 passed",
        "messages": [{"role": "user", "content": "run"}],
        "tools": WORKSPACE_TOOLS,
        "files": {"a.py": "ok\n"},
        "expect_files": {"a.py": "ok\n"},
        "max_steps": 3,
    }

    def teacher_for(line: str):
        def complete(messages, tools=None):
            if not any(message["role"] == "tool" for message in messages):
                return {"content": "", "tool_calls": [{"name": "bash", "arguments": {"command": f"echo {line}"}}]}
            return json.dumps({"answer": "ok"})

        class Teacher:
            def complete(self, messages, tools=None):
                return complete(messages, tools)

        return Teacher()

    assert generate_trajectory(task, teacher_for("nope")) is None
    assert generate_trajectory(task, teacher_for("3 passed")) is not None


def test_repeated_call_is_dropped_and_correction_updates_expect_files():
    loop = {
        "id": "loop",
        "kind": "trajectory",
        "answer": "ok",
        "messages": [{"role": "user", "content": "go"}],
        "tools": WORKSPACE_TOOLS,
        "max_steps": 4,
    }

    def repeat(messages, tools=None):
        seen = sum(1 for message in messages if message["role"] == "tool")
        if seen < 3:
            return {"content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo hi"}}]}
        return json.dumps({"answer": "ok"})

    class Repeater:
        def complete(self, messages, tools=None):
            return repeat(messages, tools)

    assert generate_trajectory(loop, Repeater()) is None

    feature = {
        "id": "feature",
        "kind": "trajectory",
        "answer": "ok",
        "correction": "also update the caller",
        "messages": [{"role": "user", "content": "add it"}],
        "tools": WORKSPACE_TOOLS,
        "files": {"a.py": "old\n"},
        "expect_files": {"a.py": "new\n"},
        "max_steps": 4,
    }

    def complete(messages, tools=None):
        if not any(message["role"] == "tool" for message in messages):
            return {"content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo FAIL"}}]}
        if not any(message["role"] == "user" and "caller" in message["content"] for message in messages):
            return json.dumps({"answer": "ok"})
        if not any(message["role"] == "tool" and message["name"] == "edit_file" for message in messages):
            return {
                "content": "",
                "tool_calls": [{
                    "name": "edit_file",
                    "arguments": {"path": "a.py", "old": "old\n", "new": "new\n"},
                }],
            }
        return json.dumps({"answer": "ok"})

    class Teacher:
        def complete(self, messages, tools=None):
            return complete(messages, tools)

    row = generate_trajectory(feature, Teacher())
    assert row is not None
    assert any(message["role"] == "user" and message["content"] == "also update the caller" for message in row["messages"])
