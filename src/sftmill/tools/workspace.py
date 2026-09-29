"""Per-episode workspace. Paths that leave the root do not run."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

TRUNCATION_MARKER = "[truncated]"
DEFAULT_MAX_OUTPUT = 2000
DEFAULT_MAX_HITS = 50

_BASH_ENV = {
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
}


def cap_output(text: str, limit: int, marker: str = TRUNCATION_MARKER) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(marker))
    return text[:keep] + marker


class Workspace:
    def __init__(self, root: str | Path, max_output: int = DEFAULT_MAX_OUTPUT, max_hits: int = DEFAULT_MAX_HITS):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_output = max_output
        self.max_hits = max_hits

    def resolve(self, path: str | None) -> Path | None:
        raw = "." if path in (None, "") else str(path)
        if ".." in Path(raw).parts:
            return None
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        root = self.root
        if resolved != root and root not in resolved.parents:
            return None
        return resolved

    def seed(self, files: dict | None) -> None:
        for path, content in (files or {}).items():
            self.write_file(str(path), "" if content is None else str(content))

    def read_text(self, path: str) -> str | None:
        resolved = self.resolve(path)
        if resolved is None or not resolved.is_file():
            return None
        return resolved.read_text()

    def call(self, name: str, arguments: dict) -> str:
        handler = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "search": self.search,
            "bash": self.bash,
        }.get(name)
        if handler is None:
            return f"error: unknown tool {name}"
        try:
            return handler(**{key: arguments.get(key) for key in arguments})
        except TypeError:
            return "error: invalid arguments"

    def list_dir(self, path: str = ".") -> str:
        resolved = self.resolve(path)
        if resolved is None or not resolved.is_dir():
            return "error: path escapes workspace"
        names = sorted(entry.name for entry in resolved.iterdir())
        return cap_output("\n".join(names), self.max_output)

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        resolved = self.resolve(path)
        if resolved is None or not resolved.is_file():
            return "error: path escapes workspace" if resolved is None else "error: not a file"
        lines = resolved.read_text().splitlines()
        start = 0 if offset in (None, "") else max(int(offset) - 1, 0)
        window = lines[start:] if limit in (None, "") else lines[start:start + int(limit)]
        return cap_output("\n".join(window), self.max_output)

    def write_file(self, path: str, content: str = "") -> str:
        resolved = self.resolve(path)
        if resolved is None or resolved == self.root:
            return "error: path escapes workspace"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text("" if content is None else str(content))
        return "ok"

    def edit_file(self, path: str, old: str = "", new: str = "") -> str:
        resolved = self.resolve(path)
        if resolved is None or not resolved.is_file():
            return "error: path escapes workspace" if resolved is None else "error: not a file"
        text = resolved.read_text()
        count = text.count(str(old))
        if count != 1:
            return "error: old string is missing" if count == 0 else "error: old string appears more than once"
        resolved.write_text(text.replace(str(old), str(new), 1))
        return "ok"

    def search(self, pattern: str = "", path: str = ".") -> str:
        if not pattern:
            return "error: invalid arguments"
        resolved = self.resolve(path)
        if resolved is None or not resolved.exists():
            return "error: path escapes workspace"
        files = [resolved] if resolved.is_file() else sorted(item for item in resolved.rglob("*") if item.is_file())
        hits: list[str] = []
        for file_path in files:
            try:
                lines = file_path.read_text().splitlines()
            except UnicodeDecodeError:
                continue
            relative = file_path.relative_to(self.root).as_posix()
            for number, line in enumerate(lines, start=1):
                if pattern in line:
                    hits.append(f"{relative}:{number}:{line}")
                    if len(hits) >= self.max_hits:
                        return cap_output("\n".join(hits), self.max_output)
        return cap_output("\n".join(hits), self.max_output)

    def _bash_argv(self, command: str) -> list[str] | None:
        try:
            argv = shlex.split(command)
        except ValueError:
            return None
        if not argv:
            return None
        if self._needs_shell(command, argv):
            return ["/bin/sh", "-c", command]
        return argv

    @staticmethod
    def _needs_shell(command: str, argv: list[str]) -> bool:
        if any(ch in command for ch in "|&;<>$`"):
            return True
        head = argv[0]
        if head in {"cd", "export", "source", "."}:
            return True
        if "=" in head:
            key = head.split("=", 1)[0]
            if key.isidentifier() or key.replace("_", "").isalnum():
                return True
        return False

    def run(self, command: str = "") -> tuple[int, str, str]:
        """Return ``(returncode, stdout, stderr)``. Setup errors use code 1."""
        if not command or self._command_escapes(command):
            return 1, "error: path escapes workspace", ""
        argv = self._bash_argv(command)
        if argv is None:
            return 1, "error: invalid arguments", ""
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=5,
                cwd=self.root,
                env=_BASH_ENV,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 1, "error: timeout", ""
        except OSError as exc:
            return 1, f"error: {exc.strerror or exc}: {argv[0]}", ""
        return completed.returncode, completed.stdout.decode(), completed.stderr.decode()

    def bash(self, command: str = "") -> str:
        code, stdout, stderr = self.run(command)
        if stdout.startswith("error:"):
            text = stdout
        elif code != 0:
            text = "error: " + stderr[-2000:]
        else:
            text = stdout
        return cap_output(text, self.max_output)

    def _command_escapes(self, command: str) -> bool:
        if ".." in command:
            return True
        try:
            tokens = shlex.split(command)
        except ValueError:
            return True
        for token in tokens:
            if token == ".." or ".." in Path(token).parts:
                return True
            if token.startswith("/"):
                resolved = Path(token).resolve()
                if resolved != self.root and self.root not in resolved.parents:
                    return True
        return False
