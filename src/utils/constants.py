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

PATH_TO_DNN_MODEL = Path(__file__).parent.parent.parent.joinpath(
    "models"
)  # path to model

PATH_TO_DATA_FOLDER = Path(__file__).parent.parent.parent.joinpath(
    "data"
)  # path to data folder

PATH_TO_VIDEOS = Path(__file__).parent.parent.parent.joinpath("videos")

PATH_TO_LOGS = Path(__file__).parent.parent.parent.joinpath("logs")

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
SOFTMAX_PARAM = 1.0

# time delay parameter
# each timestamp is time() * 1000.
# TIME_DELTA defines time diffence in 1e-3 sec.
TIME_DELTA = 50  # each TIME_DELTA * 1e-3 sec.

# barrier waiting time, depends on the start
# for safety reasons we need to assigng a large value, but control it
# otherwise one thread is broken and all application is in trouble
CAMERA_WAIT_TIME = 1.0  # 1.0 sec.
DATA_WAIT_TIME = 10.0  # 60.0 sec. Just for safety

# for ML data sampling
PROB_PARAM_ZERO = 0.05
PROB_PARAM_DISANCE = 0.025
DISTACE_LIMIT = 0.25

# which ML Model to use in the current run
ML_MODELS_AVAILABLE = ["KAN", "GB", "MLP", "HEURISTIC"]
ML_MODEL_USE = "HEURISTIC"  # possible values are listed above
