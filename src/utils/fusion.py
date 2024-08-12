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
    current_unique_frames: set[str]
    points: deque[tuple[int, str, dict[str, pd.DataFrame]]]
    fusion_results: list[tuple[int, dict[str, pd.DataFrame]]]

    def __init__(self, time_delta: int):
        """Create a new instance."""
        # save time delta between two frames
        self.time_delta = time_delta

        # to process data
        self.current_unique_frames = set()
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
        # if there is no frames, then just add them
        if len(self.points) == 0:
            self.add_elem(timestamp, camera_id, landmarks)
            return

        # if there are some elemts, then check time difference
        if abs(timestamp - self.points[0][0]) <= self.time_delta:
            # if time delay is not very big, check if we have frame of this camera
            if camera_id in self.current_unique_frames:
                # make fusion
                self.make_fusion()

                # delete old frames
                self.clear_for_camera(camera_id=camera_id)

                # append a new frame
                self.add_elem(timestamp, camera_id, landmarks)

            else:
                # just add it and wait other frames
                self.points.append((timestamp, camera_id, landmarks))
                self.current_unique_frames.add(camera_id)

        # otherwise clear container untill we do not have all frames close to the new one
        else:
            # time delay is big enough, so make fusion for current data
            self.make_fusion()

            # add a new frame keeping internal conditions of a class
            self.clear_for_timestamp(timestamp=timestamp)
            self.clear_for_camera(camera_id=camera_id)

            # save new element
            self.add_elem(timestamp, camera_id, landmarks)

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

    def add_elem(
        self, timestamp: int, camera_id: str, landmarks: dict[str, pd.DataFrame]
    ):
        """
        Insert a new element without check of conditions.

        Parameters
        ----------
        timestamp: int
            Timestamp of a new frame.
        camera_id: int
            Camera that captured a new frame.
        landmarks: dict[str, pd.DataFrame]
            Landmarks that were detected for each hand.
        """
        self.points.append((timestamp, camera_id, landmarks))
        self.current_unique_frames.add(camera_id)

    def clear_for_camera(self, camera_id: str):
        """
        Delete all elements untill camera_id is deleted.

        Parameters
        ----------
        camera_id: int
            Camera ID that must be found and deleted.
        """
        while camera_id in self.current_unique_frames:
            self.current_unique_frames.remove(self.points[0][1])
            self.points.popleft()

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
            self.current_unique_frames.remove(self.points[0][1])
            self.points.popleft()

    def clear(self):
        """Clear all internal fields."""
        self.points.clear()
        self.current_unique_frames.clear()
        self.fusion_results.clear()
