"""Command line for sftmill tasks and generate."""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from sftmill.dataset import JsonlDatasetWriter
from sftmill.generate.synthesize_tasks import synthesize_tasks
from sftmill.generate.traces import generate_trace
from sftmill.generate.trajectories import generate_trajectory
from sftmill.log import configure_logging
from sftmill.progress import ProgressReporter
from sftmill.schema import load_jsonl
from sftmill.teachers.openai_compat import OpenAICompatibleTeacher


def _teacher_from_args(args) -> OpenAICompatibleTeacher:
    return OpenAICompatibleTeacher(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=getattr(args, "timeout", 120),
        stream=getattr(args, "stream", True),
        max_tokens=getattr(args, "max_tokens", 8192),
    )


def _add_teacher_args(parser: argparse.ArgumentParser, *, max_tokens_default: int = 8192) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)


class _Discard:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def cmd_tasks(args) -> int:
    teacher = _teacher_from_args(args)
    if args.jobs > 1:
        teacher.stream_to = _Discard()
    progress = ProgressReporter("tasks", interval=args.progress_interval)
    progress.start()
    try:
        rows, shortfalls = synthesize_tasks(
            args.curriculum,
            teacher,
            args.out,
            batch_size=args.batch_size,
            jobs=args.jobs,
            progress=progress,
        )
    finally:
        progress.stop()
    print(f"wrote {len(rows)} tasks to {args.out}", flush=True)
    for message in shortfalls:
        print(message, flush=True)
    return 1 if shortfalls else 0


def _generate_one(task: dict, teacher, args):
    kind = task.get("kind", "trace")
    if args.kind in {"trace", "both"} and kind == "trace":
        return generate_trace(task, teacher)
    if args.kind in {"trajectory", "both"} and kind == "trajectory":
        return generate_trajectory(task, teacher, max_steps=args.max_steps)
    return None


def cmd_generate(args) -> int:
    import logging

    log = logging.getLogger("sftmill.generate")
    tasks = load_jsonl(args.tasks)
    teacher = _teacher_from_args(args)
    if hasattr(args, "temperature"):
        teacher.temperature = args.temperature
    if args.jobs > 1:
        teacher.stream_to = _Discard()
    selected = [
        task for task in tasks
        if (args.kind in {"trace", "both"} and task.get("kind", "trace") == "trace")
        or (args.kind in {"trajectory", "both"} and task.get("kind") == "trajectory")
    ]
    progress = ProgressReporter("generate", interval=args.progress_interval)
    progress.set(total=len(selected), processed=0, kept=0, failed=0)
    progress.start()
    written = 0
    write_lock = threading.Lock()
    workers = max(1, int(args.jobs))
    try:
        with JsonlDatasetWriter(args.out, shard_size=args.shard_size) as writer:
            def run(task: dict):
                try:
                    row = _generate_one(task, teacher, args)
                except Exception:
                    log.exception("generate failed task=%s", task.get("id"))
                    progress.bump("failed")
                    progress.bump("processed")
                    progress.set(last=task.get("id", "?"))
                    return 0
                progress.bump("processed")
                progress.set(last=task.get("id", "?"))
                if row is None:
                    return 0
                with write_lock:
                    writer.write(row)
                    count = writer.written
                progress.bump("kept")
                progress.set(written=count)
                log.info("appended %s task=%s total=%s", task.get("kind", "trace"), task.get("id"), count)
                return 1

            if workers == 1 or len(selected) <= 1:
                written = sum(run(task) for task in selected)
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
                    written = sum(pool.map(run, selected))
    finally:
        progress.stop()
    print(f"wrote {written} rows under {args.out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sftmill")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")
    parser.add_argument("--log-level", default=None, help="override log level (INFO, DEBUG, ...)")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write traces and trajectories from a teacher")
    generate.add_argument("--tasks", required=True)
    generate.add_argument("--out", required=True)
    _add_teacher_args(generate, max_tokens_default=32768)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--kind", choices=["trace", "trajectory", "both"], default="both")
    generate.add_argument("--max-steps", type=int, default=4)
    generate.add_argument("--jobs", type=int, default=3, help="teacher calls in flight (default 3)")
    generate.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="when --out is a directory, start a new part-XXXXXX.jsonl every N rows",
    )
    generate.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="seconds between progress lines on stderr (0 disables periodic reports)",
    )
    generate.set_defaults(func=cmd_generate)

    tasks = sub.add_parser("tasks", help="synthesize tasks.jsonl from a curriculum YAML")
    tasks.add_argument("--curriculum", required=True)
    tasks.add_argument("--out", required=True)
    _add_teacher_args(tasks)
    tasks.add_argument("--temperature", type=float, default=0.7)
    tasks.add_argument("--batch-size", type=int, default=5)
    tasks.add_argument("--jobs", type=int, default=3, help="categories in flight (default 3)")
    tasks.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="seconds between progress lines on stderr (0 disables periodic reports)",
    )
    tasks.set_defaults(func=cmd_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "log_level", None), getattr(args, "verbose", 0))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
