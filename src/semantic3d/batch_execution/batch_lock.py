"""Exclusive filesystem lock preventing two workers from owning one batch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType


class BatchLock:
    """Atomic O_EXCL lock; stale locks require explicit operator handling."""

    def __init__(self, lock_root: str | Path, batch_id: str) -> None:
        self.path = Path(lock_root) / f"{batch_id}.lock"
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"Batch is already locked: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle, sort_keys=True)
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "BatchLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

