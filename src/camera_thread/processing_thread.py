"""
Module gathers information from different threads and makes fusion.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event
from collections import deque
import numpy as np

# RealSense
import pyrealsense2 as rs

# to process frames


class CameraThreadRS(Thread):
    """
    Class that describes thread that makes fusion and saves the results.

    Attributes
    ----------
    stop_thread: Event
        Event to stop all threads.
    fusion_results: list[tuple[int, pd.DataFrame]]
        Resulting world coordinates of all landmarks at timestamp.
    data_source: dict[int, deque[tuple[int, np.array, np.array, rs.pyrealsense2.intrinsics]]]
        Deques where data come from.
    """

    def __init__(
        self,
        stop_thread: Event,
        threads: dict[
            int, deque[tuple[int, np.array, np.array, rs.pyrealsense2.intrinsics]]
        ],
    ):
        """
        Initialize a new instance of Thread.

        Parameters
        ----------
        """
        Thread.__init__(self)
        self.stop_thread = stop_thread
        self.sources = threads

    def is_source_empty(self):
        """Check if sources do not have any data to process and can not generate any data mode."""
        # if event is set, no generation possible
        no_generation = self.stop_thread.is_set()

        # if there is still is information in source
        is_info = False
        for source in self.sources:
            if len(self.sources[source]) > 0:
                is_info = True
                break

        return no_generation and not is_info

    def run(self):
        """Run thread and process results."""
        # current data for process
        # cameras = set()

        # we iterate until no info anymore
        while not self.is_source_empty():
            break
