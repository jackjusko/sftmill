"""Command line for sftmill tasks and generate."""

from __future__ import annotations

import argparse
import logging
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
from sftmill.teachers.openai_compat import OpenAICompatibleTeacher, TeacherPool


DEFAULT_JOBS_PER_URL = 3


def parse_teacher_endpoints(
    base_urls: list[str],
    jobs: list[int] | None,
) -> list[tuple[str, int]]:
    """Pair ``--base-url`` values with ``--jobs`` counts.

    One ``--jobs`` applies to every URL. Repeating ``--jobs`` once per URL
    sets concurrency for that server. ``--jobs`` defaults to 3 per URL.
    """
    urls = [str(url).strip() for url in base_urls if str(url).strip()]
    if not urls:
        raise ValueError("at least one --base-url is required")
    if jobs is None:
        counts = [DEFAULT_JOBS_PER_URL] * len(urls)
    elif len(jobs) == 1:
        counts = [jobs[0]] * len(urls)
    elif len(jobs) == len(urls):
        counts = list(jobs)
    else:
        raise ValueError(
            f"--jobs must be given once (applied to every --base-url) or once per "
            f"--base-url (got {len(jobs)} --jobs and {len(urls)} --base-url)"
        )
    for url, count in zip(urls, counts):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"--jobs must be a positive integer (got {count} for {url})")
    return list(zip(urls, counts))


def _teachers_from_args(args) -> tuple[list[tuple[str, int]], list[OpenAICompatibleTeacher]]:
    endpoints = parse_teacher_endpoints(args.base_url, args.jobs)
    total = sum(count for _, count in endpoints)
    stream_to = _Discard() if total > 1 else None
    teachers: list[OpenAICompatibleTeacher] = []
    for url, count in endpoints:
        for _ in range(count):
            teachers.append(
                OpenAICompatibleTeacher(
                    base_url=url,
                    model=args.model,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    timeout=getattr(args, "timeout", 120),
                    stream=getattr(args, "stream", True),
                    stream_to=stream_to,
                    max_tokens=getattr(args, "max_tokens", 8192),
                )
            )
    return endpoints, teachers


def _teacher_pool_from_args(args) -> TeacherPool:
    endpoints, teachers = _teachers_from_args(args)
    logging.getLogger("sftmill.generate").info(
        "teacher pool: %s (total jobs=%s)",
        ", ".join(f"{count} @ {url}" for url, count in endpoints),
        len(teachers),
    )
    return TeacherPool(teachers)


def _add_teacher_args(
    parser: argparse.ArgumentParser,
    *,
    max_tokens_default: int = 8192,
    jobs_help: str | None = None,
) -> None:
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        metavar="URL",
        help="OpenAI-compatible API root. Repeat for each server (same --model).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--jobs",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help=jobs_help
        or "in-flight calls for the matching --base-url (default 3). Repeat once per URL.",
    )


class _Discard:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def cmd_tasks(args) -> int:
    pool = _teacher_pool_from_args(args)
    progress = ProgressReporter("tasks", interval=args.progress_interval)
    progress.start()
    try:
        rows, shortfalls = synthesize_tasks(
            args.curriculum,
            pool,
            args.out,
            batch_size=args.batch_size,
            jobs=pool.size,
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
    log = logging.getLogger("sftmill.generate")
    tasks = load_jsonl(args.tasks)
    slots = _teacher_pool_from_args(args)
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
    workers = slots.size
    try:
        with JsonlDatasetWriter(args.out, shard_size=args.shard_size) as writer:
            def run(task: dict):
                try:
                    with slots.borrow() as teacher:
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
                with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
                    written = sum(executor.map(run, selected))
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
    _add_teacher_args(
        generate,
        max_tokens_default=32768,
        jobs_help="in-flight teacher calls for the matching --base-url (default 3). Repeat once per URL.",
    )
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--kind", choices=["trace", "trajectory", "both"], default="both")
    generate.add_argument("--max-steps", type=int, default=4)
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
    _add_teacher_args(
        tasks,
        jobs_help="in-flight categories for the matching --base-url (default 3). Repeat once per URL.",
    )
    tasks.add_argument("--temperature", type=float, default=0.7)
    tasks.add_argument("--batch-size", type=int, default=5)
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
    if args.command in {"tasks", "generate"}:
        try:
            parse_teacher_endpoints(args.base_url, args.jobs)
        except ValueError as exc:
            parser.error(str(exc))
    configure_logging(getattr(args, "log_level", None), getattr(args, "verbose", 0))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
