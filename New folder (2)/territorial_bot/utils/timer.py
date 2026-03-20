"""
Timing utilities for Territorial.io bot.
"""

from __future__ import annotations

import time


class GameTimer:
    """
    Lightweight wall-clock timer for episodes and environment steps.
    """

    def __init__(self) -> None:
        """
        Initialize the timer and start it immediately.
        """

        self.start_time = time.time()

    def start(self) -> None:
        """
        Start or restart the timer.
        """

        self.start_time = time.time()

    def elapsed_seconds(self) -> float:
        """
        Return elapsed wall-clock time in seconds.

        Returns:
            float: Elapsed seconds.
        """

        return time.time() - self.start_time

    def elapsed_ms(self) -> float:
        """
        Return elapsed wall-clock time in milliseconds.

        Returns:
            float: Elapsed milliseconds.
        """

        return self.elapsed_seconds() * 1000.0

    def reset(self) -> None:
        """
        Reset the timer start point to the current time.
        """

        self.start()
