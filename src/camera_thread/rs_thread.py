"""
Module contains implementation of thread using depth-camera Intel RealSense.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event, Lock, Barrier, BrokenBarrierError
from time import time
import numpy as np
import pandas as pd

# realsense camera
from camera_thread.camera import camera
import pyrealsense2 as rs

# models
from hand_recognition.HolisticLandmarker import HolisticLandmarker
from hand_recognition.hand_recognizer import extract_landmarks
from utils.utils import make_video
from utils.constants import DATA_WAIT_TIME, CAMERA_WAIT_TIME


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
    target: tuple[int, str, mp.tasks.vision.HolisticLandmarkerResult, np.ndarray, rs.pyrealsense2.intrinsics]
        A place to save the result (timestamp, cameara_id, Left and Right hands)
    lock: Lock
        A locker to controll multithread access to the target-attribute
    barrier: Barrier
        A barrier to force different cameras to work simutaneously
    """

    close_event: Event
    frames: list[np.ndarray]
    target: tuple[
        int, str, dict[str, pd.DataFrame], np.ndarray, rs.pyrealsense2.intrinsics
    ] | tuple[None, None, None, None, None] = None, None, None, None, None
    barrier: Barrier
    data_barrier: Barrier
    locker: Lock

    def __init__(
        self,
        camera_name: str,
        camera_id: str,
        close_event: Event,
        barrier: Barrier,
        data_barrier: Barrier,
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
        self.data_barrier = data_barrier

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        # define landmarker
        holistic_landmarker = HolisticLandmarker()

        # start frames
        while not self.close_event.is_set():
            # get data from picture
            color_frame = self.camera.take_picture_and_return_color()
            depth_frame = self.camera.get_last_depth_frame()
            intrinsics = self.camera.get_last_intrinsics()
            # get time stamp
            time_stamp = int(time() * 1000)

            # give time for other threads but not to much
            if not self.syncronize():
                print("Camera-Barrier is broken. Processing impossible.")
                self.close_event.set()
                break

            # run mediapipe
            mp_results = holistic_landmarker.process_image(
                holistic_landmarker, color_frame
            )
            detected_hands = extract_landmarks(mp_results=mp_results)

            # give time for other threads but not to much
            if not self.syncronize():
                print("Camera-Barrier is broken. Processing impossible.")
                self.close_event.set()
                break

            # for debugging only !!!!
            # self.frames.append(color_frame)
            # draw_landmarks_holistics(color_frame, mp_results.left_hand_landmarks)
            # draw_landmarks_holistics(color_frame, mp_results.right_hand_landmarks)

            # if there is something, add it
            with self.locker:
                if len(detected_hands) > 0:
                    self.target = (
                        time_stamp,
                        self.camera.device_id,
                        detected_hands,
                        depth_frame,
                        intrinsics,
                    )
                else:
                    self.target = None, None, None, None, None

            # show that thread has provided data
            try:
                self.data_barrier.wait(timeout=DATA_WAIT_TIME)
            except BrokenBarrierError:
                self.close_event.set()
                continue

        # stop camera
        self.camera.stop()

        # create video that was actually caprured
        self.make_video()

        # report finish !!!
        print(f"Thread {self.camera.device_id} is stopped")

    def make_video(self):
        """Create video from frames."""
        make_video(name=self.camera.device_id, frames=self.frames)

    def get_frame(
        self,
    ) -> (
        tuple[int, str, dict[str, pd.DataFrame], np.ndarray, rs.pyrealsense2.intrinsics]
        | tuple[None, None, None, None, None]
    ):  # type: ignore
        """Return latest frame possible."""
        with self.locker:
            return self.target

    def syncronize(self) -> bool:
        """
        Syncronize all threads in one point.

        Returns
        -------
        bool
            True - threads are syncronized
            False - error
        """
        try:
            self.barrier.wait(timeout=CAMERA_WAIT_TIME)
            return True
        except BrokenBarrierError:
            return False

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
