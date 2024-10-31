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
    model: Any
    scaler: StandardScaler
    camera_id: str

    def __init__(self, camera_id: str):
        """Crea a new instance from file."""
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
