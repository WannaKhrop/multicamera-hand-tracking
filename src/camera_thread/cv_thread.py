"""
Module contains thread implementation to work with OpenCV package.
In case of OpenCV it's necessary do identify a method to extract depth information from picture.

Author: Ivan Khrop
Date: 31.07.2024
"""

# import basic libraries
from threading import Thread, Event
from time import time
import cv2
import numpy as np


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
    stream: bool = False
        Make a stream. For testing purposes.
    """

    def __init__(
        self,
        camera_id: int,
        close_event: Event,
        target: list[tuple[int, np.array]],
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
