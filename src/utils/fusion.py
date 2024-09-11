"""
Module contains class that performs fusion operation for different threads.

Author: Ivan Khrop
Date: 09.08.2024
"""

from collections import deque
import pandas as pd
from utils.geometry import landmarks_fusion
from utils.constants import SOFTMAX_PARAM


class DataMerger:
    """
    Class makes fusion of landmarks from different threads.

    Attributes
    ----------
    time_delta: int > 0
        All frames are different no more than time_delta for timestamps.
    current_unique_frames: set[int]
        Set that contains unique cameras ID that will be fused in the next step.
    points: deque[tuple[int, str, dict[str, pd.DataFrame]]]
        Landmarks of different cameras [timestamp, camera_id, landmarks].
    fusion_results: list[tuple[int, dict[str, pd.DataFrame]]]
        Results of fusion (timestamp, landmarks).
    """

    time_delta: int
    points: deque[tuple[int, str, dict[str, pd.DataFrame]]]
    fusion_results: list[tuple[int, dict[str, pd.DataFrame]]]

    def __init__(self, time_delta: int):
        """Create a new instance."""
        # save time delta between two frames
        self.time_delta = time_delta

        # to process data
        self.points = deque()

        # resulting coordinates
        self.fusion_results = list()

    def add_time_frame(
        self, timestamp: int, camera_id: str, landmarks: dict[str, pd.DataFrame]
    ):
        """
        Process a new frame.

        Parameters
        ----------
        timestamp: int
            Timestamp of a new frame.
        camera_id: int
            Camera that captured a new frame.
        landmarks: dict[str, pd.DataFrame]
            Landmarks that were detected for each hand.
        """
        # no frames = just add
        if len(self.points) == 0:
            self.points.append((timestamp, camera_id, landmarks))
        # if it's early frame it can be added
        elif (
            timestamp < self.points[0][0]
            and (self.points[0][0] - timestamp) <= self.time_delta
        ):
            self.points.append((timestamp, camera_id, landmarks))
        # if there are some elemts, then check time difference
        # clear container untill we do not have all frames close to the new one
        elif timestamp - self.points[0][0] > self.time_delta:
            self.points.append((timestamp, camera_id, landmarks))
            self.clear_for_timestamp(timestamp=timestamp)
        # if frame timestamp is fine
        elif timestamp - self.points[0][0] <= self.time_delta:
            self.points.append((timestamp, camera_id, landmarks))
        else:
            return

        # fusion
        self.make_fusion()

    def make_fusion(self):
        """Make fusion for current state."""
        # for each and make fusion
        result = dict()

        # go over all points and get the number of hands
        hands = set()
        for frame in self.points:
            # gather all hand available
            hands.update(list(frame[2].keys()))

        # for each hand make a fusion
        timestamp = 0
        for hand in hands:
            # save world coordinates here
            world_coordinates = list()

            # gather information from all the frames of different cameras
            for frame in self.points:
                if hand in frame[2]:
                    timestamp = max(timestamp, frame[0])
                    world_coordinates.append(frame[2][hand])

            # make fusion and save results
            result[hand] = landmarks_fusion(
                world_coordinates=world_coordinates, softmax_const=SOFTMAX_PARAM
            )

        # save the final result
        self.fusion_results.append((timestamp, result))

    def clear_for_timestamp(self, timestamp: int):
        """
        Delete all elements untill all timestamps including a new one differ no more than time delay.

        Parameters
        ----------
        timestamp: int
            Point in time that must be satisfied.
        """
        while (
            len(self.points) > 0
            and abs(timestamp - self.points[0][0]) > self.time_delta
        ):
            self.points.popleft()

    def clear(self):
        """Clear all internal fields."""
        self.points.clear()
        self.fusion_results.clear()
