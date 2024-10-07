"""
Module contains class that uses trained neural network to convert normalized landmarks in world coodinates landmarks.

Author: Ivan Khrop
Date: 23.09.2024
"""

from keras.models import load_model, Sequential
import tensorflow as tf
import numpy as np
from utils.constants import PATH_TO_DNN_MODEL
from utils.utils import TimeChecker, CustomLoss


class MedapipeWorldTransformer:
    model: Sequential

    def __init__(self):
        """Crea a new instance from file."""
        self.model = load_model(
            filepath=PATH_TO_DNN_MODEL, custom_objects={"CustomLoss": CustomLoss}
        )

    @TimeChecker
    def __call__(self, features: np.ndarray) -> tf.Tensor:
        return self.predict(features)

    @tf.function(jit_compile=True)
    def predict(self, features: np.ndarray) -> tf.Tensor:
        """
        Predict world coordinates.

        Parameters
        ----------
        features: np.ndarray
            Results of mediapipe as normalized coordinates.
        Returns
        -------
        tf.Tensor
            Tensor with depths [Nx21].
        """
        results = self.model(features)

        return results
