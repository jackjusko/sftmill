import json
from types import SimpleNamespace

import pytest

from sftmill.cli import (
    _Discard,
    _teachers_from_args,
    build_parser,
    cmd_generate,
    main,
    parse_teacher_endpoints,
)
from sftmill.teachers.openai_compat import TeacherPool


def test_parser_subcommands():
    parser = build_parser()
    subs = {action.dest for action in parser._subparsers._actions if hasattr(action, "choices") and action.choices}
    assert subs == {"command"}
    command_parser = next(
        a for a in parser._subparsers._actions if hasattr(a, "choices") and a.choices
    )
    assert set(command_parser.choices) == {"tasks", "generate"}


@pytest.mark.parametrize("argv", [
    ["train", "--config", "x.yaml"],
    ["smoke", "--config", "x.yaml"],
    ["bench", "--config", "x.yaml"],
    ["probe"],
    ["shell", "--config", "x.yaml"],
])
def test_unknown_commands_fail(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code != 0


def test_parse_teacher_endpoints_defaults_jobs_per_url():
    assert parse_teacher_endpoints(["http://a/v1"], None) == [("http://a/v1", 3)]
    assert parse_teacher_endpoints(["http://a/v1", "http://b/v1"], None) == [
        ("http://a/v1", 3),
        ("http://b/v1", 3),
    ]


def test_parse_teacher_endpoints_one_jobs_applies_to_every_url():
    assert parse_teacher_endpoints(["http://a/v1", "http://b/v1"], [4]) == [
        ("http://a/v1", 4),
        ("http://b/v1", 4),
    ]


def test_parse_teacher_endpoints_pairs_jobs_with_urls():
    assert parse_teacher_endpoints(["http://a/v1", "http://b/v1"], [4, 2]) == [
        ("http://a/v1", 4),
        ("http://b/v1", 2),
    ]


def test_parse_teacher_endpoints_rejects_mismatch_and_non_positive():
    with pytest.raises(ValueError, match="once per"):
        parse_teacher_endpoints(["http://a/v1", "http://b/v1"], [4, 2, 1])
    with pytest.raises(ValueError, match="positive"):
        parse_teacher_endpoints(["http://a/v1"], [0])


def test_parser_accepts_repeated_base_url_and_jobs():
    parser = build_parser()
    args = parser.parse_args([
        "generate",
        "--tasks", "t.jsonl",
        "--out", "out",
        "--model", "qwen3.8-27b",
        "--base-url", "http://127.0.0.1:8080/v1", "--jobs", "4",
        "--base-url", "http://127.0.0.1:8081/v1", "--jobs", "2",
    ])
    assert parse_teacher_endpoints(args.base_url, args.jobs) == [
        ("http://127.0.0.1:8080/v1", 4),
        ("http://127.0.0.1:8081/v1", 2),
    ]


def test_teachers_from_args_creates_one_slot_per_job():
    args = SimpleNamespace(
        base_url=["http://a/v1", "http://b/v1"],
        jobs=[4, 2],
        model="qwen3.8-27b",
        api_key=None,
        temperature=0.5,
        timeout=10,
        stream=False,
        max_tokens=16,
    )
    endpoints, teachers = _teachers_from_args(args)
    assert endpoints == [("http://a/v1", 4), ("http://b/v1", 2)]
    assert [teacher.base_url for teacher in teachers] == ["http://a/v1"] * 4 + ["http://b/v1"] * 2
    assert {teacher.model for teacher in teachers} == {"qwen3.8-27b"}
    assert all(isinstance(teacher.stream_to, _Discard) for teacher in teachers)


def test_single_job_keeps_stdout_streaming():
    args = SimpleNamespace(
        base_url=["http://a/v1"],
        jobs=[1],
        model="m",
        api_key=None,
        temperature=0.2,
        timeout=10,
        stream=True,
        max_tokens=16,
    )
    _, teachers = _teachers_from_args(args)
    assert len(teachers) == 1
    assert teachers[0].stream_to is None


def test_teacher_pool_borrow_returns_each_slot():
    first, second = object(), object()
    pool = TeacherPool([first, second])
    assert pool.size == 2
    with pool.borrow() as a:
        with pool.borrow() as b:
            assert {a, b} == {first, second}


def test_main_rejects_mismatched_jobs():
    with pytest.raises(SystemExit):
        main([
            "generate",
            "--tasks", "t.jsonl",
            "--out", "out",
            "--model", "m",
            "--base-url", "http://a/v1",
            "--base-url", "http://b/v1",
            "--jobs", "4",
            "--jobs", "2",
            "--jobs", "1",
        ])


def test_cmd_generate_borrows_a_teacher_per_url(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.jsonl"
    rows = [
        {"id": f"t{i}", "kind": "trace", "messages": [{"role": "user", "content": "q"}]}
        for i in range(6)
    ]
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    seen: list[str] = []

    def fake_trace(task, teacher):
        seen.append(teacher.base_url)
        return {
            "kind": "trace",
            "task_id": task["id"],
            "messages": list(task["messages"]) + [{"role": "assistant", "content": "ok"}],
        }

    monkeypatch.setattr("sftmill.cli.generate_trace", fake_trace)
    args = build_parser().parse_args([
        "generate",
        "--tasks", str(tasks_path),
        "--out", str(tmp_path / "out.jsonl"),
        "--model", "m",
        "--kind", "trace",
        "--base-url", "http://a/v1", "--jobs", "2",
        "--base-url", "http://b/v1", "--jobs", "1",
        "--no-stream",
        "--progress-interval", "0",
    ])
    assert cmd_generate(args) == 0
    assert len(seen) == 6
    assert set(seen) == {"http://a/v1", "http://b/v1"}
