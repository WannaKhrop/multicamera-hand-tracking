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
    unique_frames: set[tuple[int, str]]
    fusion_results: list[tuple[int, dict[str, pd.DataFrame]]]

    def __init__(self, time_delta: int):
        """Create a new instance."""
        # save time delta between two frames
        self.time_delta = time_delta

        # to process data
        self.points = deque()

        # unique frames
        self.unique_frames = set()

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
        # check if we already have this frame
        if (timestamp, camera_id) in self.unique_frames:
            return

        # check if this frame is in the past
        if len(self.points) > 0 and self.points[0][0] - timestamp > self.time_delta:
            return

        # add frame and update set and sort frames
        self.points.append((timestamp, camera_id, landmarks))
        self.unique_frames.add((timestamp, camera_id))
        self.points = deque(sorted(self.points, key=lambda frame: frame[0]))

        # adjust frames for fusion
        self.clear_for_timestamp()

        # fusion
        self.make_fusion()

    def make_fusion(self):
        """Make fusion for current state."""
        # debug
        """
        for point in self.points:
            print(point[0], point[1])
        print(60 * "=")
        """

        # go over all points and get the number of hands
        hands = set(["Left", "Right"])
        # for each hand make fusion
        result = dict()

        # for each hand make a fusion
        timestamp = 0
        for hand in hands:
            # save world coordinates here
            world_coordinates = list()

            # gather information from all the frames of different cameras
            for frame_timestamp, _, frame in self.points:
                if hand in frame:
                    timestamp = max(timestamp, frame_timestamp)
                    world_coordinates.append(frame[hand])

            # make fusion and save results
            if len(world_coordinates) > 0:
                result[hand] = landmarks_fusion(
                    world_coordinates=world_coordinates, softmax_const=SOFTMAX_PARAM
                )

        # save the final result
        self.fusion_results.append((timestamp, result))

    def clear_for_timestamp(self):
        """Delete all elements untill all timestamps differ no more than time delay."""
        while (
            len(self.points) > 0
            and abs(self.points[-1][0] - self.points[0][0]) > self.time_delta
        ):
            timestamp, camera_id, _ = self.points[0]

            # remove frame and delete from set
            self.points.popleft()
            self.unique_frames.remove((timestamp, camera_id))

    def clear(self):
        """Clear all internal fields."""
        self.points.clear()
        self.unique_frames.clear()
        self.fusion_results.clear()
