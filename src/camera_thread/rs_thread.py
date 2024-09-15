"""
Module contains implementation of thread using depth-camera Intel RealSense.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event, Lock
from time import time, sleep
import pandas as pd
import numpy as np

# realsense camera
from camera_thread.camera import camera
import pyrealsense2 as rs

# deque collection for frames
from collections import deque

# models
from hand_recognition.HolisticLandmarker import HolisticLandmarker
from hand_recognition.Landmarker import Landmarker
from utils.utils import make_video


class CameraThreadRS(Thread):
    """
    Class that describes PyRealSense Thread.

    Attributes
    ----------
    camera_name: str
        Name of a camera that will take pictures for this thread
    camera_id: str
        ID of a camera that will take pictures for this thread
    close_event: Event
        Event to stop the thread
    target: deque[tuple[int, str, dict[str, pd.DataFrame]]]
        A place to save the result (timestamp, cameara_id, Left and Right hands)
    use_holistics: bool = False
        If we need to use holisics model.
    use_async: bool = False
        If we use async processing in real time.
    """

    camera_name: str
    camera_id: str
    close_event: Event
    frames: deque[tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]]
    target: deque[tuple[int, str, dict[str, pd.DataFrame]]]
    use_holistics: bool
    use_async: bool
    locker: Lock

    def __init__(
        self,
        camera_name: str,
        camera_id: str,
        close_event: Event,
        target: deque[tuple[int, str, dict[str, pd.DataFrame]]],
        use_holistics: bool = False,
        use_async: bool = False,
    ):
        """Initialize a new instance of RS-Thread for a camera."""
        Thread.__init__(self)

        # init fields
        self.camera = camera(camera_name, camera_id)
        self.close_event = close_event
        self.frames: deque[
            tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]
        ] = deque()
        self.capture_target = target
        self.use_holistics = use_holistics
        self.use_async = use_async

        # init locker in case of multithreding
        self.locker = Lock()

    def get_name(self) -> str:
        """
        Get the name of a thread.

        Returns
        -------
        str:
            Name of the current thread
        """
        return "Camera #{}".format(self.camera.device_id)

    def get_camera(self) -> camera:
        """
        Get reference to the camera of this thread.

        Returns
        -------
        camera:
            Camera object that captured images for this thread
        """
        return self.camera

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        while True:
            # get data from picture
            color_frame = self.camera.take_picture_and_return_color()
            depth_frame = self.camera.get_last_depth_frame()
            intrinsics = self.camera.get_last_intrinsics()
            # get time stamp
            time_stamp = int(time() * 1000)

            # save the results of this frame
            frame = (
                time_stamp,
                self.camera.device_id,
                color_frame,
                depth_frame,
                intrinsics,
            )

            self.add_new_frame(frame)

            # give time for other threads
            sleep(0.05)

            # if threads are stopped
            if self.close_event.is_set():
                break

        # stop camera
        self.camera.stop()

        # if we use async, no need to check
        if not self.use_async:
            # define mode
            if self.use_holistics:
                holistic_landmarker = HolisticLandmarker()
                processed = holistic_landmarker.process_frames(self.frames)
            else:
                landmarker = Landmarker()
                processed = landmarker.process_frames(self.frames)

            # store frames
            self.capture_target.extend(processed)

    def make_video(self):
        """Create video from frames."""
        make_video(self.frames)

    def add_new_frame(
        self, frame: tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]
    ):
        """Put new frame in container."""
        with self.locker:
            self.frames.append(frame)

    def get_frame(
        self, idx
    ) -> tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]:
        """Return latest frame possible."""
        with self.locker:
            try:
                return self.frames[idx]
            except IndexError:
                return self.frames[-1]

    @classmethod
    def returnCameraIndexes(cls) -> list[tuple[str, str]]:
        """
        Identify the list of available cameras.

        Returns
        -------
        list[tuple[str, int]]:
            List with identificators of available cameras (name, ID).
        """
        arr = []

        context = rs.context()
        for device in context.devices:
            arr.append(
                (
                    device.get_info(rs.camera_info.name),
                    device.get_info(rs.camera_info.serial_number),
                )
            )

        return arr
