"""
Module desribes transformer that converts points 
from camera coordinate system to world coordinate system.

Author: Ivan Khrop
Date: 09.08.2024
"""
# import some important data
import numpy as np
import glob

from utils.constants import PATH_TO_TRANSFORMATION, NUMPY_FILE_EXT


class CoordinateTransformer:
    """
    Singleton that performs all the transformation operations.

    Attributes
    ----------
    transformations: dict[int, np.ndarray]
    """

    # fields of transformer
    transformations: dict[str, np.array]

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(CoordinateTransformer, cls).__new__(
                cls, *args, **kwargs
            )
            cls._instance.transformations = dict()

            # find all .csv files in data directory
            file_pattern = PATH_TO_TRANSFORMATION.joinpath(f"*{NUMPY_FILE_EXT}")

            # Find all CSV files in the directory
            np_files = glob.glob(str(file_pattern))

            # Read each CSV file and store the DataFrames in a list
            for file in np_files:
                # get camera id as string from name of a file
                camera_id = str(file).split("\\")[-1].rstrip(NUMPY_FILE_EXT)
                # read matrix
                matrix = np.load(file)

                # save matrix
                cls._instance.transformations[camera_id] = matrix

        return cls._instance

    def camera_to_world(self, camera_id: str, points: np.ndarray) -> np.ndarray:
        """
        Apply transformation to points in camera coordinate system.

        Parameters
        ----------
        camera_id: str
            Camera that transformation should be applied for.
        camera_coords: np.ndarray
            Data in camera coordinates.

        Returns
        -------
        np.ndarray
            Result after transformation = (R * camera_coords + t) in world coordinates
        """
        # check if we have th
        assert (
            camera_id in self.transformations
        ), f"Transformation matrix for camera {camera_id} is absent."

        # small transformation to add translation
        x_data = np.vstack([points.T, np.ones((1, points.shape[0]))])

        # apply transformation
        y_data = np.dot(self.transformations[camera_id], x_data)
        y_data = y_data[:3]  # we need only first 3 columns because the 4-th is ones !!!

        return y_data.T

    def world_to_camera(self, camera_id: str, points: np.ndarray) -> np.ndarray:
        # no need
        return np.zeros(1)
