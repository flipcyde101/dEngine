"""Readable live status for a single cell, including validation and output."""

from datetime import datetime, timedelta
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Progress:
    def __init__(self, label):
        self.label = label
        self.started = time.monotonic()
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.status = "active"
        self.stage = "preparing data"
        self.step = 1
        self.detail = "opening and checking data"
        try:
            self.zone = ZoneInfo("America/Denver")
        except ZoneInfoNotFoundError:
            self.zone = datetime.now().astimezone().tzinfo

    def __enter__(self):
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()
        self._emit("started")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed.set()
        self.worker.join()
        self.status = "failed" if exc_type else "complete"
        self.detail = str(exc_type.__name__) if exc_type else "output hashes recorded"
        if not exc_type:
            self.step = 4
        self._emit("finished")

    def _heartbeat(self):
        while not self.closed.wait(20):
            self._emit("heartbeat")

    def _emit(self, event):
        with self.lock:
            print(
                f"{self.label} | {self.status} | {self.stage} | "
                f"steps {self.step - (self.status != 'complete')}/4 complete | "
                f"{time.monotonic() - self.started:.1f}s elapsed | {event} | {self.detail}",
                flush=True,
            )

    def update(self, step, stage, detail="estimating after a measured epoch"):
        with self.lock:
            self.step = step
            self.stage = stage
            self.detail = detail
        self._emit("update")

    def epoch(self, completed, total, samples, elapsed_training):
        remaining = (total - completed) * elapsed_training / completed
        eta = datetime.now(self.zone) + timedelta(seconds=remaining)
        self.update(
            2, "training and validation",
            f"epochs {completed}/{total}, {samples * completed} examples processed, "
            f"{samples * completed / max(elapsed_training, 0.001):.1f} examples/s, "
            f"training remaining ~{remaining:.0f}s (through {eta:%H:%M:%S %Z}), "
            "final evaluation and output follow",
        )
