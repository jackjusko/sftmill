"""Short default system prompt for Alice instruct traces."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_IDENTITY_PATH = Path(__file__).resolve().parents[2] / "configs" / "identity" / "alice.txt"

TEACHER_OPEN_INSTRUCT = (
    "Do not mention a repo, workspace, or files unless the user provided them. "
    "Do not emit tool calls or tool XML."
)


@lru_cache(maxsize=1)
def default_system() -> str:
    return _IDENTITY_PATH.read_text(encoding="utf-8").strip()


def has_system(messages: list | None) -> bool:
    return any(message.get("role") == "system" for message in messages or [])


def prepend_default_system(messages: list[dict] | None) -> list[dict]:
    """Put the identity system first unless the row already has a system turn."""
    existing = list(messages or [])
    if has_system(existing):
        return existing
    text = default_system()
    if not text:
        return existing
    return [{"role": "system", "content": text}, *existing]


def teacher_system(student_system: str | None) -> str:
    if student_system and student_system.strip():
        return f"{student_system.strip()}\n\n{TEACHER_OPEN_INSTRUCT}"
    return TEACHER_OPEN_INSTRUCT
