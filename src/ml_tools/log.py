import logging
import time
from typing import Optional


class Timer:
    def __init__(
        self,
        name: str = "Timer",
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        interval: str = 's'
    ):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.level = level

        self._interval_map = {
            'ms': 0.001,
            's': 1,
            'm': 60,
            'h': 3600
        }
        self.time_interval = interval
        self.interval_factor = self._interval_map[interval]

        self.start: float = -1.0
        self.last_interval: float = -1.0
        self.end: float = -1.0

    def __enter__(self):
        self.start = time.time()
        self.logger.log(self.level, f"{self.name}: __start__")
        return self
    
    def __exit__(self, *args):
        interval = time.time() - self.start
        msg = f"{self.name}: __end__ {interval * self.interval_factor:.3f}{self.time_interval} elapsed"
        self.logger.log(self.level, msg)

    def __call__(self, msg: str = ""):
        curr_time = time.time()
        if self.last_interval == -1.0:
            self.last_interval = curr_time

        interval = curr_time - self.start
        since_last = curr_time - self.last_interval
        msg = f"{self.name}: {msg} tot: {interval * self.interval_factor:.3f}{self.time_interval} " \
            + f"lap: {since_last * self.interval_factor:.3f}{self.time_interval}"
        self.logger.log(self.level, msg)
        self.last_interval = curr_time