"""
Module contains class that uses trained neural network to convert normalized landmarks in world coodinates landmarks.

Author: Ivan Khrop
Date: 23.09.2024
"""
import tensorflow as tf
import numpy as np
from utils.constants import PATH_TO_DNN_MODEL
from utils.utils import TimeChecker  # , CustomLoss

# from keras.models import load_model
import joblib
from typing import Any


class MedapipeWorldTransformer:
    model: Any

    def __init__(self):
        """Crea a new instance from file."""
        PATH_TO_MODEL = PATH_TO_DNN_MODEL.joinpath(
            "multi_output_gb_regressor.joblib"
        )  # joinpath("mediapipe_world_model.h5")
        # self.model = load_model(
        #    filepath=PATH_TO_MODEL, custom_objects={"CustomLoss": CustomLoss}
        # )
        self.model = joblib.load(PATH_TO_MODEL)

    @TimeChecker
    def __call__(self, features: np.ndarray) -> tf.Tensor | np.ndarray:
        # return self.predict(features)
        return self.model.predict(features)

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
