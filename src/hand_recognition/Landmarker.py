"""
Module contains class of landmarker to process images.
This class is aimed at work with sequences of images. Each image is counted as independent.

Author: Ivan Khrop
Date: 18.08.2024
"""
# basic imports
import mediapipe as mp
from utils.constants import PATH_TO_MODEL

import numpy as np
import pandas as pd
import pyrealsense2 as rs

# to process frames
from hand_recognition.hand_recognizer import convert_to_camera_coordinates, hand_to_df
from utils.coordinate_transformer import CoordinateTransformer
from utils.geometry import assign_visability

# define types
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class Landmarker:
    """
    This is a singleton class that creates one example of landmarker that can be used
    to detect handlandmarks at all images.

    Attribures
    ----------
    landmarker: HandLandmarker.
    """

    def __init__(self, *args, **kwargs):
        """Create a new instance."""

        # usage of hands - model !!!
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=PATH_TO_MODEL),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def process_image(
        self,
        image: np.ndarray,
        timestamp: int,
    ) -> mp.tasks.vision.HolisticLandmarkerResult:  # type: ignore
        """
        Return results of the detection for an image.

        Parameters
        ----------
        image: np.ndarray
            Image that must be processed.
        timestamp: int
            Timestamp of this frame.

        Returns
        -------
        detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
            Results of mediapipe
        """
        # convert image to mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        return self.landmarker.detect_for_video(mp_image, timestamp)

    def process_frames(
        self,
        frames: list[tuple[int, int, np.array, np.array, rs.pyrealsense2.intrinsics]],
    ) -> list[tuple[int, int, dict[str, pd.DataFrame]]]:
        # define transformer
        transformer = CoordinateTransformer()

        # process each frame and save those ones that have detected hands
        detected_results = list()
        for timestamp, camera_id, color_frame, depth_frame, intrinsics in frames:
            # process image and get information
            mp_results = self.process_image(color_frame, timestamp)
            detected_hands = convert_to_camera_coordinates(
                mp_results, depth_frame, intrinsics
            )

            # check if it's empty then MediaPipe has not found hand on this frame
            if len(detected_hands) == 0:
                continue

            # assign convert to world coordinates and assign visibility to each frame
            for hand in detected_hands:
                # world coords
                world_coords = transformer.camera_to_world(
                    camera_id=camera_id, points=detected_hands[hand]
                )
                world_coords = hand_to_df(world_coords)

                # convert to pd.DataFrame
                detected_hands[hand] = hand_to_df(detected_hands[hand])

                # assign visibility
                assign_visability(df_landmarks=detected_hands[hand])

                # save results
                world_coords["visibility"] = detected_hands[hand].loc[
                    world_coords.index, "visibility"
                ]
                detected_hands[hand] = world_coords

            # save detected results
            detected_results.append((timestamp, camera_id, detected_hands))

        return detected_results
