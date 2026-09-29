from sftmill.tools.python_sandbox import run_python


def test_sandbox_prints_stdout():
    assert run_python("print(1 + 1)").strip() == "2"


def test_sandbox_times_out():
    assert run_python("import time\ntime.sleep(5)\n", timeout=0.2) == "error: timeout"


def test_sandbox_reports_errors():
    assert run_python("raise SystemExit(1)").startswith("error:")
