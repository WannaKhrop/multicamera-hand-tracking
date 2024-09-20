"""
Module contains singleton class of landmarker to process stream of images using holistic solution.

Author: Ivan Khrop
Date: 31.07.2024
"""

import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import pandas as pd

from hand_recognition.hand_recognizer import (
    convert_to_camera_coordinates_holistic,
    draw_landmarks_holistics,
)
from utils.coordinate_transformer import CoordinateTransformer
from utils.utils import make_video, TimeChecker

from typing import Iterable


class HolisticLandmarker:
    """
    This is a singleton class that creates one example of landmarker that can be used
    to detect handlandmarks at all images.

    Attribures
    ----------
    landmarker: mp.tasks.vision.HolisticLandmarker
    """

    def __init__(self, *args, **kwargs):
        """Create a new instance."""
        # usage of holistics solution
        model = mp.solutions.holistic.Holistic(
            static_image_mode=False,  # we process static images, not stream
            model_complexity=2,  # can be ignored, influences pose-model
            enable_segmentation=False,  # no need to generate segmentation mask
            refine_face_landmarks=False,  # no interest in face landmarks in this solution
            min_detection_confidence=0.6,  # confidence in detection
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
            Image that must be processed

        Returns
        -------
        detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
            Results of mediapipe
        """
        return self.landmarker.process(image)

    def process_frames(
        self,
        frames: Iterable[
            tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]
        ],
    ) -> Iterable[tuple[int, str, dict[str, pd.DataFrame]]]:
        # define transformer
        transformer = CoordinateTransformer()

        # process each frame and save those ones that have detected hands
        detected_results = list()
        for timestamp, camera_id, color_frame, depth_frame, intrinsics in frames:
            # process image and get information
            mp_results = self.process_image(self, image=color_frame)

            draw_landmarks_holistics(color_frame, mp_results.left_hand_landmarks)
            draw_landmarks_holistics(color_frame, mp_results.right_hand_landmarks)

            detected_hands = convert_to_camera_coordinates_holistic(
                mp_results, depth_frame, intrinsics
            )

            # check if it's empty then MediaPipe has not found hand on this frame
            if len(detected_hands) == 0:
                continue

            # assign convert to world coordinates and assign visibility to each frame
            axis = ["x", "y", "z"]
            for hand in detected_hands:
                # world coords
                detected_hands[hand][axis] = transformer.camera_to_world(
                    camera_id=camera_id, points=detected_hands[hand][axis].values
                )

            # save detected results
            detected_results.append((timestamp, camera_id, detected_hands))

        make_video(frames=frames)

        return detected_results
