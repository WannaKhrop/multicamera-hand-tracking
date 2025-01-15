"""
Module contains implementation of thread-data-collector using depth-camera Intel RealSense.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event
import numpy as np

# realsense camera
from camera_thread.camera import camera
import pyrealsense2 as rs

# models
from hand_recognition.HolisticLandmarker import HolisticLandmarker
from hand_recognition.hand_recognizer import to_numpy_ndarray
from utils.utils import thread_safe
from utils.constants import (
    DISTACE_LIMIT,
    CAMERA_RESOLUTION_WIDTH,
    CAMERA_RESOLUTION_HEIGHT,
)


@thread_safe
def report(name: str, n_samples: int):
    print(f"Thread {name} collected {n_samples} frames.")


class MLCameraThreadRS(Thread):
    """
    Class that describes PyRealSense Thread.

    Attributes
    ----------
    thread_cam: camera
        Camera that captures frames for this thread.
    close_event: Event
        Event to stop the thread.
    target: list[tuple[np.ndarray, np.ndarray]]
        A place to save the result (timestamp, cameara_id, Left and Right hands).
    """

    thread_cam: camera
    close_event: Event
    target: list[tuple[np.ndarray, np.ndarray]]

    def __init__(
        self,
        camera_name: str,
        camera_id: str,
        close_event: Event,
    ):
        """Initialize a new instance of RS-Thread for a camera."""
        Thread.__init__(self)

        # init fields
        self.thread_cam = camera(camera_name, camera_id)
        self.close_event = close_event
        self.target = list()

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        # create landmarker
        landmarker = HolisticLandmarker()

        # run camera
        while not self.close_event.is_set():
            # get data from picture
            color_frame = self.thread_cam.take_picture_and_return_color()
            depth_frame = self.thread_cam.get_last_depth_frame()
            intrinsics = self.thread_cam.get_last_intrinsics()

            # process frame
            mp_results = landmarker.process_image(landmarker, color_frame)

            # for each hand get depths and
            if mp_results.left_hand_landmarks is not None:
                landmarks = to_numpy_ndarray(mp_results.left_hand_landmarks)
                # get depth data
                features, depths = MLCameraThreadRS.process_hand(
                    landmarks, depth_frame, intrinsics
                )
                depth_max, depth_min = np.max(depths), np.min(depths)
                # check if depths are well assigned
                if (depths > 1e-3).all() and (depth_max - depth_min) < DISTACE_LIMIT:
                    self.target.append((features, depths))

            # for each hand get depths and
            if mp_results.right_hand_landmarks is not None:
                landmarks = to_numpy_ndarray(mp_results.right_hand_landmarks)
                # get depth data
                features, depths = MLCameraThreadRS.process_hand(
                    landmarks, depth_frame, intrinsics
                )
                depth_max, depth_min = np.max(depths), np.min(depths)
                # check if depths are well assigned
                if (depths > 1e-3).all() and (depth_max - depth_min) < DISTACE_LIMIT:
                    self.target.append((features, depths))

            # report
            report(self.thread_cam.device_id, len(self.target))

        # stop camera
        self.thread_cam.stop()

    @staticmethod
    def process_hand(
        landmarks: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: rs.pyrealsense2.intrinsics,
    ):
        # get depth data
        rel_depths = landmarks[:, 2]
        depths = MLCameraThreadRS.get_camera_coordinates(
            landmarks, depth_frame, intrinsics
        )

        return (rel_depths, depths)

    @staticmethod
    def get_camera_coordinates(
        landmarks: np.ndarray,
        depth_frame: np.ndarray,
        intrinsics: rs.pyrealsense2.intrinsics,
    ) -> np.ndarray:
        # storage
        depth_data = list()

        for landmark in landmarks:
            # get pixels
            x_pixel = int(CAMERA_RESOLUTION_WIDTH * landmark[0])
            y_pixel = int(CAMERA_RESOLUTION_HEIGHT * landmark[1])
            # get depth
            depth = camera.get_depth(x_pixel, y_pixel, depth_frame=depth_frame)
            # get camera coordinates
            coordinates = camera.get_coordinates_for_depth(
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                depth=depth,
                intrinsics=intrinsics,
            )
            depth_data.append(coordinates[2])

        return np.array(depth_data)

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
