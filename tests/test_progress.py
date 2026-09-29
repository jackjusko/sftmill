import io
import time

from sftmill.progress import ProgressReporter


def test_progress_reporter_emits_on_stop():
    stream = io.StringIO()
    reporter = ProgressReporter("test", interval=60.0, stream=stream)
    reporter.set(total=10, done=3)
    reporter.stop()
    text = stream.getvalue()
    assert "[sftmill test]" in text
    assert "done=3" in text
    assert "total=10" in text


def test_progress_reporter_periodic_emit():
    stream = io.StringIO()
    reporter = ProgressReporter("tick", interval=0.05, stream=stream)
    reporter.set(n=1)
    reporter.start()
    time.sleep(0.12)
    reporter.stop()
    assert stream.getvalue().count("[sftmill tick]") >= 2
