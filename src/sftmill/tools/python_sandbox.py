"""Timeout-limited Python tool. The child does not inherit CUDA or the parent env."""

from __future__ import annotations

import subprocess
import sys
import tempfile

_RUNNER = """
import sys
code = sys.stdin.read()
namespace = {"__name__": "__main__"}
exec(compile(code, "<tool>", "exec"), namespace)
"""


def run_python(code: str, timeout: float = 5.0, cwd: str | None = None) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=code.encode(),
            capture_output=True,
            timeout=timeout,
            cwd=cwd or tempfile.gettempdir(),
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                "CUDA_VISIBLE_DEVICES": "",
            },
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error: timeout"
    stdout = completed.stdout.decode()
    if completed.returncode != 0:
        stderr = completed.stderr.decode()[-2000:]
        return "error: " + stderr
    return stdout
