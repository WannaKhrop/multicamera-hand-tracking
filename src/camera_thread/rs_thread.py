"""
Module contains implementation of thread using depth-camera Intel RealSense.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event
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
    target: deque[tuple[int, int, np.array, np.array, rs.pyrealsense2.intrinsics]]
        A place to save the result (timestamp, cameara_id, color_frame, depth_frame, intrinsics)
    process_images: bool = False
        If thread must process sequence of images itself.
    use_holistics: bool = False
        If we need to use holisics model.
    """

    def __init__(
        self,
        camera_name: str,
        camera_id: str,
        close_event: Event,
        target: deque[tuple[int, str, dict[str, pd.DataFrame]]],
        use_holistics: bool = False,
    ):
        """
        Initialize a new instance of RS-Thread for a camera.

        Parameters
        ----------
        camera_name: str
        Name of a camera that will take pictures for this thread
        camera_id: int
            ID of a camera that will take pictures for this thread
        close_event: Event
            Event to stop the thread
        target: deque[tuple[int, str, dict[str, pd.DataFrame]]]
            A place to save the result (timestamp, color_frame, depth_frame, intrinsics)
        use_holistics: bool = False
            If we need to use holisics model.
        """
        Thread.__init__(self)
        self.camera = camera(camera_name, camera_id)
        self.close_event = close_event
        self.frames: deque[
            tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]
        ] = deque()
        self.capture_target = target
        self.use_holistics = use_holistics

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
            self.frames.append(
                (
                    time_stamp,
                    self.camera.device_id,
                    color_frame,
                    depth_frame,
                    intrinsics,
                )
            )

            # give time for other threads
            sleep(0.02)

            # if threads are stopped
            if self.close_event.is_set():
                break

        # stop camera
        self.camera.stop()

        # define mode
        if self.use_holistics:
            holistic_landmarker = HolisticLandmarker()
            processed = holistic_landmarker.process_frames(self.frames)
        else:
            landmarker = Landmarker()
            processed = landmarker.process_frames(self.frames)

        # store frames
        self.capture_target.extend(processed)

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
