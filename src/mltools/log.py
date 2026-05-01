"""Logging helpers."""

import logging
from time import time


class Timer:
    """Log elapsed time for named operations."""

    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        level: int = logging.DEBUG,
    ):
        self.name = name
        self.logger = logger
        self.level = level

        self.start: float | None = None
        self.elapsed: float = 0
        self.end: float | None = None

    def __call__(self, message: str):
        """Log elapsed time since the previous checkpoint."""
        elapsed = time() - self.elapsed
        self.logger.log(self.level, self._format(f"{message}: {elapsed} seconds"))
        self.elapsed = elapsed

    def __enter__(self):
        """Start the timer."""
        self.start = time()
        self.elapsed = self.start
        return self

    def __exit__(self, *args):
        """Log the final elapsed time."""
        self.end = time()
        interval = self.end - self.start
        self.logger.log(self.level, self._format(f"END: {interval} seconds"))

    def _format(self, message: str) -> str:
        return f"{self.name}: {message} "
