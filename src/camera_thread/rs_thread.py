"""
Module contains implementation of thread using depth-camera Intel RealSense.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event, Lock, Barrier, BrokenBarrierError
from time import time
import pandas as pd
import numpy as np

# realsense camera
from camera_thread.camera import camera
import pyrealsense2 as rs

# models
from hand_recognition.HolisticLandmarker import HolisticLandmarker
from hand_recognition.hand_recognizer import convert_to_camera_coordinates_holistic
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
    target: tuple[int, str, dict[str, pd.DataFrame]]
        A place to save the result (timestamp, cameara_id, Left and Right hands)
    lock: Lock
        A locker to controll multithread access to the target-attribute
    barrier: Barrier
        A barrier to force different cameras to work simutaneously
    """

    close_event: Event
    frames: list[np.ndarray]
    target: tuple[int, str, dict[str, pd.DataFrame]]
    barrier: Barrier
    locker: Lock

    def __init__(
        self, camera_name: str, camera_id: str, close_event: Event, barrier: Barrier
    ):
        """Initialize a new instance of RS-Thread for a camera."""
        Thread.__init__(self)

        # init fields
        self.camera = camera(camera_name, camera_id)
        self.frames = list()

        # in case of multithreding
        self.locker = Lock()
        self.close_event = close_event
        self.barrier = barrier

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        # define landmarker
        holistic_landmarker = HolisticLandmarker()

        # start frames
        while True:
            # get data from picture
            color_frame = self.camera.take_picture_and_return_color()
            depth_frame = self.camera.get_last_depth_frame()
            intrinsics = self.camera.get_last_intrinsics()
            # get time stamp
            time_stamp = int(time() * 1000)

            # run mediapipe
            mp_results = holistic_landmarker.process_image(
                holistic_landmarker, color_frame
            )
            # detect hands
            detection_results = convert_to_camera_coordinates_holistic(
                mp_results, depth_frame, intrinsics
            )

            # for debugging only !!!!
            """
            draw_landmarks_holistics(color_frame, mp_results.left_hand_landmarks)
            draw_landmarks_holistics(color_frame, mp_results.right_hand_landmarks)
            self.frames.append(color_frame)
            """

            # give time for other threads but not to much
            try:
                self.barrier.wait(timeout=1.0)
            except BrokenBarrierError:
                assert (
                    False
                ), "Barrier broken, proceeding without synchronization is impossible."

            # if there is something, add it
            if len(detection_results) > 0:
                self.add_new_frame(
                    (time_stamp, self.camera.device_id, detection_results)
                )

            # if threads are stopped
            if self.close_event.is_set():
                break

        # stop camera
        self.camera.stop()

        # for debugging only !!!!
        """
        self.make_video()
        """

    def make_video(self):
        """Create video from frames."""
        make_video(name=self.camera.device_id, frames=self.frames)

    def add_new_frame(self, frame: tuple[int, str, dict[str, pd.DataFrame]]):
        """Put new frame in container."""
        with self.locker:
            self.target = frame

    def get_frame(
        self,
    ) -> tuple[int, str, dict[str, pd.DataFrame]] | tuple[None, None, None]:
        """Return latest frame possible."""
        with self.locker:
            if hasattr(self, "target"):
                return self.target
            else:
                return None, None, None

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
