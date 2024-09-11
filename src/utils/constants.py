"""
Module contains imporant constants that are used over all modules in the project.

Author: Ivan Khrop
Date: 06.08.2024
"""

# paths to different folders
# assuming that modules main.py and find_transformation.py are running

from pathlib import Path

# PATH_TO_TRANSFORMATION = "..\\transformations\\"
PATH_TO_TRANSFORMATION = Path(__file__).parent.parent.parent.joinpath(
    "transformations"
)  # transformations

# PATH_TO_MODEL = "..\\models\\hand_landmarker.task"
PATH_TO_MODEL = (
    Path(__file__)
    .parent.parent.parent.joinpath("models")
    .joinpath("hand_landmarker.task")
)  # path to model

PATH_TO_DATA_FOLDER = Path(__file__).parent.parent.parent.joinpath(
    "data"
)  # path to data folder

PATH_TO_VIDEOS = Path(__file__).parent.parent.parent.joinpath("videos")

# numpy format
NUMPY_FILE_EXT = ".npy"

# depth_data.get_data() returns integer values in 10^(-3) meter
# at the same time depth_data.get_depth(x, y) returns the value in meters
# so we need to scale it !!!
SCALE_FACTOR = 1_000.0

# camera resolution data
CAMERA_RESOLUTION_WIDTH = 1920
CAMERA_RESOLUTION_HEIGHT = 1080

# softmax parameter to make fusion
SOFTMAX_PARAM = 50.0

# time delay parameter
# each timestamp is time() * 1000.
# TIME_DELTA defines time diffence in 1e-3 sec.
TIME_DELTA = 10  # each TIME_DELTA * 1e-3 sec.
