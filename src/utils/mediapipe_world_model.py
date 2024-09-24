"""
Module contains class that uses trained neural network to convert normalized landmarks in world coodinates landmarks.

Author: Ivan Khrop
Date: 23.09.2024
"""

from keras.models import load_model, Sequential
import numpy as np
from utils.constants import PATH_TO_DNN_MODEL


class MedapipeWorldTransformer:
    model: Sequential

    def __init__(self):
        """Crea a new instance from file."""
        self.model = load_model(filepath=PATH_TO_DNN_MODEL)

    def predict(self, features: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """
        Predict world coordinates.

        Parameters
        ----------
        features: np.ndarray
            Results of mediapipe as normalized coordinates.
        shape: tuple[int, int]
            Output shape.
        Returns
        -------
        np.ndarray
            Result as matrix [21x3].
        """
        results = self.model.predict(features)

        return results.reshape(shape)
