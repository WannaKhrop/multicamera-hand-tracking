"""
Module contains singleton class of landmarker to process stream of images using holistic solution.

Author: Ivan Khrop
Date: 31.07.2024
"""

import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import pandas as pd

from hand_recognition.hand_recognizer import extract_landmarks
from utils.utils import TimeChecker

from typing import Iterable


class HolisticLandmarker:
    """
    This is a singleton class that creates one example of landmarker that can be used
    to detect handlandmarks at all images.

    Attribures
    ----------
    landmarker: mp.tasks.vision.HolisticLandmarker
    """

    landmarker: mp.tasks.vision.HolisticLandmarker  # type: ignore

    def __init__(self, *args, **kwargs):
        """Create a new instance."""
        # usage of holistics solution
        model = mp.solutions.holistic.Holistic(
            static_image_mode=False,  # we process static images, not stream
            model_complexity=2,  # can be ignored, influences pose-model
            enable_segmentation=False,  # no need to generate segmentation mask
            refine_face_landmarks=False,  # no interest in face landmarks in this solution
            min_detection_confidence=0.7,  # confidence in detection
        )

        self.landmarker = model

    @TimeChecker
    def process_image(
        self, image: np.ndarray
    ) -> mp.tasks.vision.HolisticLandmarkerResult:  # type: ignore
        """
        Return results of the detection for an image.

        Parameters
        ----------
        image: np.ndarray
            Image that must be processed.

        Returns
        -------
        detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
            Results of mediapipe.
        """
        return self.landmarker.process(image)

    def process_frames(
        self,
        frames: Iterable[
            tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]
        ],
    ) -> Iterable[tuple[int, str, dict[str, pd.DataFrame]]]:
        """
        Processes a sequence of frames to detect hands and extract landmarks.

        Parameters
        ----------
        frames: Iterable[tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]]
            An iterable of tuples, where each tuple contains:
                - timestamp (int): The timestamp of the frame.
                - camera_id (str): The identifier of the camera.
                - color_frame (np.ndarray): The color frame image.
                - depth_frame (np.ndarray): The depth frame image.
                - intrinsics (rs.pyrealsense2.intrinsics): The camera intrinsics.

        Returns
        -------
        Iterable[tuple[int, str, dict[str, pd.DataFrame]]]:
            An iterable of tuples, where each tuple contains:
                - timestamp (int): The timestamp of the frame.
                - camera_id (str): The identifier of the camera.
                - detected_hands (dict[str, pd.DataFrame]): A dictionary containing detected hand landmarks.
        """
        # process each frame and save those ones that have detected hands
        detected_results = list()
        for timestamp, camera_id, color_frame, depth_frame, _ in frames:
            # process image and get information
            mp_results = self.process_image(self, image=color_frame)

            detected_hands = extract_landmarks(
                mp_results=mp_results, depth_frame=depth_frame
            )

            # check if it's empty then MediaPipe has not found hand on this frame
            if len(detected_hands) == 0:
                continue

            # save detected results
            detected_results.append((timestamp, camera_id, detected_hands))

        return detected_results
