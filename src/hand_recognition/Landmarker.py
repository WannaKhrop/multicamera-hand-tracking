"""
Module contains singleton class of landmarker to process images.

Author: Ivan Khrop
Date: 31.07.2024
"""

import mediapipe as mp

# define types
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# path to model
PATH_TO_MODEL = "../models/hand_landmarker.task"


class Landmarker:
    """
    This is a singleton class that creates one example of landmarker that can be used
    to detect handlandmarks at all images.

    Attribures
    ----------
    landmarker: HandLandmarker
        One single instance of this class. Construction of object always returns reference
        to this one instance.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.
        """
        if not hasattr(cls, "instance"):
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=PATH_TO_MODEL),
                num_hands=2,
                running_mode=VisionRunningMode.IMAGE,
            )

            cls.instance = HandLandmarker.create_from_options(options)

        return cls.instance
