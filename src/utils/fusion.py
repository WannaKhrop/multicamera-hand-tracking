"""
Module contains class that performs fusion operation for different threads.

Author: Ivan Khrop
Date: 09.08.2024
"""

from collections import deque
from threading import Lock
import pandas as pd
from utils.geometry import landmarks_fusion
from utils.constants import SOFTMAX_PARAM
from utils.utils import TimeChecker, write_logs
from camera_thread.camera_frame import CameraFrame


class DataMerger:
    """
    Class for fusing landmarks from different camera frames.

    Attributes
    ----------
    time_delta : int
        Maximum allowed time difference between frames for fusion.
    unique_frames : set[tuple[int, str]]
        Set of unique camera IDs and timestamps to detect repeating frames.
    points : deque[CameraFrame]
        Queue of camera frames containing landmarks.
    fusion_results : list[CameraFrame]
        List of fused camera frames.
    locker : Lock
        Lock to control access to shared resources.

    Methods
    -------
    add_time_frame(camera_frame: CameraFrame)
        Adds a new camera frame for processing.
    make_fusion()
        Performs fusion of landmarks from different camera frames.
    get_latest_result() -> tuple[int, dict[str, pd.DataFrame]] | tuple[None, None]
        Retrieves the latest fusion result.
    clear_for_timestamp()
        Removes frames until all timestamps differ by no more than the time delta.
    clear()
        Clears all internal fields.
    write_logs()
        Writes logs to a file.
    """

    time_delta: int
    points: deque[CameraFrame]
    unique_frames: set[tuple[int, str]]
    fusion_results: list[CameraFrame]
    locker: Lock

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

        # locker
        self.locker = Lock()

    def add_time_frame(self, camera_frame: CameraFrame):
        """
        Add a new frame into the merger.

        Parameters
        ----------
        camera_frame: CameraFrame
            Frame after processing having landmarks in world coordinates.
        """
        # unwrap data
        timestamp, camera_id, landmarks, _ = camera_frame.as_tuple()
        # process data
        with self.locker:
            # check if we already have this frame
            if (timestamp, camera_id) in self.unique_frames:
                return

            # check if this frame is in the past
            if (
                len(self.points) > 0
                and self.points[0].timestamp - timestamp > self.time_delta
            ):
                return

            # add frame and update set and sort frames
            self.points.append(
                CameraFrame(timestamp, camera_id, landmarks, intrinsics=None)
            )
            self.unique_frames.add((timestamp, camera_id))
            self.points = deque(sorted(self.points, key=lambda frame: frame.timestamp))

            # adjust frames for fusion
            self.clear_for_timestamp()

    @TimeChecker
    def make_fusion(self):
        """Make fusion for current state."""
        # debug
        for point in self.points:
            print(point.timestamp, point.camera_id)
        print(60 * "=")

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
            for data in self.points:
                # unwrap frame
                frame_timestamp, _, frame, _ = data.as_tuple()

                # process hands
                if hand in frame:
                    timestamp = max(timestamp, frame_timestamp)
                    world_coordinates.append(frame[hand])

            # make fusion and save results
            if len(world_coordinates) > 0:
                result[hand] = landmarks_fusion(
                    world_coordinates=world_coordinates, softmax_const=SOFTMAX_PARAM
                )

        # save the final result
        camera_frame = CameraFrame(
            timestamp=timestamp,
            camera_id="Fusion",
            landmarks=result,
            intrinsics=None,
        )
        self.fusion_results.append(camera_frame)

    def get_latest_result(
        self,
    ) -> tuple[int, dict[str, pd.DataFrame]] | tuple[None, None]:
        """Get the latest merger result."""
        with self.locker:
            if len(self.fusion_results) > 0:
                frame_timestamp, _, hands_dict, _ = self.fusion_results[-1].as_tuple()
                return frame_timestamp, hands_dict
            else:
                return None, None

    def clear_for_timestamp(self):
        """Delete all elements untill all timestamps differ no more than time delay."""
        while (
            len(self.points) > 0
            and abs(self.points[-1].timestamp - self.points[0].timestamp)
            > self.time_delta
        ):
            timestamp, camera_id, _, _ = self.points[0].as_tuple()

            # remove frame and delete from set
            self.points.popleft()
            self.unique_frames.remove((timestamp, camera_id))

    def clear(self):
        """Clear all internal fields."""
        self.points.clear()
        self.unique_frames.clear()
        self.fusion_results.clear()

    def write_logs(self):
        """Write logs to the file."""
        write_logs(frames=self.fusion_results, camera_id="Fusion")
