"""
Module contains class that uses trained neural network to convert normalized landmarks in world coodinates landmarks.

Author: Ivan Khrop
Date: 23.09.2024
"""
import tensorflow as tf
import numpy as np
from utils.constants import PATH_TO_DNN_MODEL, ML_MODEL_USE, ML_MODELS_AVAILABLE
from utils.utils import TimeChecker

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from typing import Any
from kan import KAN
import torch
import joblib


class MedapipeWorldTransformer:
    """
    A transformer class for handling different machine learning models used in
    multi-camera hand tracking. This class supports various models such as KAN,
    MLP, Gradient Boosting, and a heuristic approach for depth reconstruction.

    Attributes
    ----------
    model : Any
        The machine learning model used for predictions.
    scaler : StandardScaler
        The scaler used for feature normalization (currently commented out).
    camera_id : str
        The identifier for the camera being used.

    Methods
    -------
    __init__(camera_id: str)
        Initializes the transformer with the specified camera ID and loads the
        appropriate model based on the global ML_MODEL_USE variable.
    __call__(features: np.ndarray) -> tf.Tensor | np.ndarray
        Makes predictions using the loaded model based on the provided features.
    heuristic(features: np.ndarray) -> np.ndarray
        Applies a heuristic algorithm for depth reconstruction based on the provided features.
    predict(features: np.ndarray) -> tf.Tensor
        Predicts world coordinates using the loaded TensorFlow model.
    """

    model: Any
    scaler: StandardScaler
    camera_id: str

    def __init__(self, camera_id: str):
        """Create a new instance from file."""
        self.camera_id = camera_id
        # check
        assert (
            ML_MODEL_USE in ML_MODELS_AVAILABLE
        ), "There is no ML Model that is specified"
        # path
        basic_path = PATH_TO_DNN_MODEL.joinpath(camera_id)
        # choose a model
        match ML_MODEL_USE:
            case "KAN":
                # KAN
                self.model = KAN.loadckpt(basic_path.joinpath("mark"))
                self.model.eval()
                # self.scaler = joblib.load(filename=basic_path.joinpath("scaler.joblib"))
            case "MLP":
                # tensorflow
                self.model = load_model(filepath=basic_path.joinpath(f"{camera_id}.h5"))
            case "GB":
                # gradient boosting
                self.model = joblib.load(
                    filename=basic_path.joinpath(f"{camera_id}.joblib")
                )
            case "HEURISTIC":
                pass

    @TimeChecker
    def __call__(self, features: np.ndarray) -> tf.Tensor | np.ndarray:
        # choose a model
        match ML_MODEL_USE:
            case "KAN":
                # get KAN prediction
                with torch.no_grad():
                    # self.scaler.transform(features)
                    x = torch.from_numpy(features).float()
                    predict = self.model(x)
                return predict.numpy()
            case "MLP":
                # tensorflow
                return self.predict(features)
            case "GB":
                # gradient boosting
                return self.model.predict(features)
            case "HEURISTIC":
                return self.heuristic(features)

    def heuristic(self, features: np.ndarray) -> np.ndarray:
        """
        Apply a heuristic algorithm for depth reconstruction.
        This method processes the input features to reconstruct depths using a heuristic approach.
        It separates the input features into relative depths and absolute depths, adjusts the depths
        based on a threshold, and then updates the relative depths based on the minimum depth found.

        Parameters
        ----------
        features : np.ndarray
            A 2D array of shape (n_hands, 2 * n_features) where the first half represents
            relative depths and the second half represents absolute depths.

        Returns
        -------
        np.ndarray
            A 2D array of reconstructed depths with the same shape as the input features.
        """
        # extract depths
        data = np.hsplit(features, 2)
        rel_depths, depths = data[0], data[1]
        depths = np.where(depths <= 1e-3, 100.0, depths)
        # get min depth bu more than 0.0
        min_depth = np.min(depths, axis=1, keepdims=True)
        # get argmin
        argmin_mask = np.argmin(np.abs(depths - min_depth), axis=1, keepdims=True)
        # update_relative depths
        rel_depths = (
            1.0 + rel_depths - np.take_along_axis(rel_depths, argmin_mask, axis=1)
        )
        # save new depths
        return rel_depths * min_depth

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
