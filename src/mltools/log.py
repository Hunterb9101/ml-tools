import logging
from time import time
from typing import Optional

class Timer:
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        level: int = logging.DEBUG
    ):
        self.name = name
        self.logger = logger
        self.level = level

        self.start: Optional[int] = None
        self.elapsed: Optional[int] = None
        self.end: Optional[int] = None

    def __call__(self, message: str):
        elapsed = time() - self.elapsed
        self.logger.log(self.level, self._format(f"{message}: {elapsed} seconds"))
        self.elapsed = elapsed

    def __enter__(self):
        self.start = time()
        self.elapsed = self.start
        return self

    def __exit__(self, *args):
        self.end = time()
        interval = self.end - self.start
        self.logger.log(self.level, self._format(f"END: {interval} seconds"))

    def _format(self, message: str) -> str:
        return f"{self.name}: {message} "
