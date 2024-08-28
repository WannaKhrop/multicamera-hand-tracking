"""
Module contains singleton class of landmarker to process single images using holistic solution.

Author: Ivan Khrop
Date: 31.07.2024
"""

import mediapipe as mp


class HolisticLandmarkerSingleton:
    """
    This is a singleton class that creates one example of landmarker that can be used
    to detect handlandmarks at all images.

    Attribures
    ----------
    landmarker: mp.tasks.vision.HolisticLandmarker
        One single instance of this class. Construction of object always returns reference
        to this one instance.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance.
        """
        if not hasattr(cls, "instance"):
            # usage of holistics solution
            model = mp.solutions.holistic.Holistic(
                static_image_mode=True,  # we process static images, not stream
                model_complexity=2,  # can be ignored, influences pose-model
                enable_segmentation=False,  # no need to generate segmentation mask
                refine_face_landmarks=False,  # no interest in face landmarks in this solution
                min_detection_confidence=0.6,  # confidence in detection
            )

            cls.instance = model

        return cls.instance
