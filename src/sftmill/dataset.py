"""JSONL datasets: append while generating, stream while training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

_HARNESS_SUFFIXES = ("-alice", "-openai", "-claude_code", "-novel")


def group_key(row: dict) -> str:
    """Original question id. Harness copies of one task share this key."""
    if row.get("group_id"):
        return str(row["group_id"])
    task_id = str(row.get("task_id") or row.get("id") or "")
    for suffix in _HARNESS_SUFFIXES:
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def split_by_group(
    rows: list[dict],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Put every harness sibling of a question on the same side of the split."""
    fraction = min(1.0, max(0.0, float(val_fraction)))
    train: list[dict] = []
    val: list[dict] = []
    for row in rows:
        key = group_key(row)
        digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < fraction:
            val.append(row)
        else:
            train.append(row)
    return train, val


def iter_jsonl_file(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_jsonl_dataset(path: str | Path) -> Iterator[dict]:
    """Yield rows from a ``.jsonl`` file or every ``*.jsonl`` shard in a directory."""
    root = Path(path)
    if root.is_dir():
        files = sorted(root.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no .jsonl shards under {root}")
        for shard in files:
            yield from iter_jsonl_file(shard)
        return
    if not root.is_file():
        raise FileNotFoundError(path)
    yield from iter_jsonl_file(root)


def count_jsonl_dataset(path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl_dataset(path))


class JsonlDatasetWriter:
    """Append one row at a time. Directories become numbered shards."""

    def __init__(self, path: str | Path, *, shard_size: int = 1000):
        self.root = Path(path)
        self.shard_size = max(1, int(shard_size))
        self.written = 0
        self._shard_index = 0
        self._shard_rows = 0
        self._handle = None
        self._single_file = self.root.suffix == ".jsonl"
        if self._single_file:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.root.open("a", encoding="utf-8")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self._open_shard()

    def _shard_path(self) -> Path:
        return self.root / f"part-{self._shard_index:06d}.jsonl"

    def _open_shard(self) -> None:
        if self._handle is not None:
            self._handle.close()
        path = self._shard_path()
        self._handle = path.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        if self._handle is None:
            raise RuntimeError("writer is closed")
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        self.written += 1
        if self._single_file:
            return
        self._shard_rows += 1
        if self._shard_rows >= self.shard_size:
            self._shard_index += 1
            self._shard_rows = 0
            self._open_shard()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> JsonlDatasetWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()
