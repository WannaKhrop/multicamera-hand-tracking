"""
Module contains implementation of two types of threads.
Thread for OpenCV and for RealSense. Usability depends on available cameras.
If it's a depth camera => use RealSense, otherwise OpenCV.
In case of OpenCV it's necessary do identify a method to extract depth informatio from picture.

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic libraries
from threading import Thread, Event
from time import time
import cv2
import numpy as np

# realsense camera
from camera import camera
import pyrealsense2 as rs

# import mediapipe helpers
import hand_recognizer
from mediapipe.tasks.python.components.containers import NormalizedLandmark


class CameraThreadCV(Thread):
    """
    Class that describes a Computer Vision Thread.

    Attributes
    ----------
    camera_id: int
        ID of a camera that will take pictures for this thread
    close_event: Event
        Event to stop the thread
    target: list[tuple[int, np.array]]
        A place to save the result (timestamp, frame)
    landmarker: mediapipe.tasks.vision.HandLandmarker
        Landmarker that is used by this thread
    stream: bool = False
        Make a stream. For testing purposes.
    """

    def __init__(
        self,
        camera_id: int,
        close_event: Event,
        target: list[tuple[int, np.array]],
        path_to_model: str,
        stream: bool = False,
    ):
        """
        Initialize a new instance of CV-Thread for a camera.

        Parameters
        ----------
        camera_id: int
            ID of a camera that will take pictures for this thread
        close_event: Event
            Event to stop the thread
        target: list[tuple[int, np.array]]
            A place to save the result (timestamp, frame)
        path_to_model: str
            Path to file *.task for mediapipe-hands
        stream: bool = False
            Make a stream. For testing purposes.
        """
        Thread.__init__(self)
        self.camera_id = camera_id
        self.close_event = close_event
        self.capture_target = target
        self.stream = stream
        self.landmarker = hand_recognizer.create_landmarker(path_to_model)

    def get_name(self) -> str:
        """
        Get the name of a thread.

        Returns
        -------
        str:
            Name of the current thread
        """
        return "Camera #{}".format(self.camera_id)

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        video = cv2.VideoCapture(self.camera_id)

        # for testing purposes
        if self.stream:
            cv2.namedWindow(self.get_name())

        while video.isOpened():
            ret, frame = video.read()
            time_stamp = int(time() * 1000)

            self.capture_target.append((time_stamp, self.camera_id, frame))

            # for testing purposes
            if self.stream:
                cv2.imshow(self.get_name(), frame)

            if self.close_event.is_set() or not ret:
                break

        video.release()

        # for testing purposes
        if self.stream:
            cv2.destroyWindow(self.get_name())

    @classmethod
    def returnCameraIndexes(cls) -> list[int]:
        """
        Identify the list of available cameras.
        Up to 10 cameras.

        Returns
        -------
        list[int]:
            List with identificators of available cameras.
        """
        arr = []

        for index in range(10):
            cap = cv2.VideoCapture(index)
            ret, _ = cap.read()
            if ret:
                arr.append(index)
                cap.release()

        return arr


# define usefull constants
CAMERA_RESOLUTION_WIDTH = 1920
CAMERA_RESOLUTION_HEIGHT = 1080


class CameraThreadRS(Thread):
    """
    Class that describes PyRealSense Thread.

    Attributes
    ----------
    camera_name: str
        Name of a camera that will take pictures for this thread
    camera_id: int
        ID of a camera that will take pictures for this thread
    close_event: Event
        Event to stop the thread
    target: list[tuple[int, np.array, dict[str, np.ndarray]]]
        A place to save the result (timestamp, frame, dict(hand, coordinates))
    landmarker: mediapipe.tasks.vision.HandLandmarker
        Landmarker that is used by this thread
    """

    def __init__(
        self,
        camera_name: str,
        camera_id: int,
        close_event: Event,
        target: list[tuple[int, np.array]],
        path_to_model: str,
    ):
        """
        Initialize a new instance of CV-Thread for a camera.

        Parameters
        ----------
        camera_name: str
        Name of a camera that will take pictures for this thread
        camera_id: int
            ID of a camera that will take pictures for this thread
        close_event: Event
            Event to stop the thread
        target: list[tuple[int, np.array, dict[str, np.ndarray]]]
            A place to save the result (timestamp, frame, dict(hand, coordinates))
        path_to_model: str
            Path to file *.task for mediapipe-hands
        """
        Thread.__init__(self)
        self.camera = camera(camera_name, camera_id)
        self.close_event = close_event
        self.capture_target = target
        self.landmarker = hand_recognizer.create_landmarker(path_to_model)

    def get_name(self) -> str:
        """
        Get the name of a thread.

        Returns
        -------
        str:
            Name of the current thread
        """
        return "Camera #{}".format(self.camera.device_id)

    def run(self):
        """
        Run the thread. Take pictures and save the results.
        """
        while True:
            frame = self.camera.take_picture_and_return_color()
            time_stamp = int(time() * 1000)

            # RealSense Camera Thread must process image immediatelly because each time the camera takes a frame
            # it also captures depth information. This depth information will be lost as soon as a new frame is captures.
            # So, we have to process image inside of this thread.

            results_of_mediapipe = hand_recognizer.process_image(frame, self.landmarker)
            hands = {}
            for hand in results_of_mediapipe.handedness:
                # get index and hand
                idx = hand[0].index
                name = hand[0].category_name

                # get hand world landmarks
                world_landmarks = hand_recognizer.to_numpy_ndarray(
                    results_of_mediapipe.hand_world_landmarks[idx]
                )

                # get normalized landmarks
                landmarks = hand_recognizer.to_numpy_ndarray(
                    results_of_mediapipe.hand_landmarks[idx]
                )

                # get the closest point to the camera according to z-axis
                closest_point_idx = hand_recognizer.HandLandmark(
                    np.argmin(landmarks[:, 2])
                )
                closest_point = self.extract_camera_coordinates(
                    results_of_mediapipe.hand_landmark[idx][closest_point_idx]
                )

                # make the closest point a new center of coordinates
                hand_with_new_origin = hand_recognizer.change_origin(
                    closest_point_idx, world_landmarks
                )

                # add the real world coordinates to the camera coordinates
                # save result for hand
                hands[name] = closest_point + hand_with_new_origin

            # save the results of this frame
            self.capture_target.append((time_stamp, frame, hands))

            if self.close_event.is_set():
                break

    @classmethod
    def returnCameraIndexes(cls) -> list[tuple[str, int]]:
        """
        Identify the list of available cameras.
        Up to 10 cameras.

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

    def extract_camera_coordinates(self, landmark: NormalizedLandmark) -> np.ndarray:
        """
        Transform mediapipe landmark into a np.ndarray extracting depth information.
        The landmark must be taken from "hand_landmark"-option of a detection.
        See the corresponding test.

        Parameters
        ----------
        landmark: NormalizedLandmark
            Landmark in mediapipe.
        cam: camera
            Camera that took the picture and is able to return depth.

        Returns
        -------
        numpy.ndarray
            Vector [x, y, z] of a landmark in camera coordinates
        """
        # identify a  pixel
        x_pixel = landmark.x * CAMERA_RESOLUTION_WIDTH
        y_pixel = landmark.y * CAMERA_RESOLUTION_HEIGHT

        # get camera coodinates
        x, y, z = tuple(self.camera.get_depth_data_from_pixel(x_pixel, y_pixel))

        # return the result
        return np.array([x, y, z])
