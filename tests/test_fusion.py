import os
import sys
import numpy as np
import pandas as pd

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from utils.geometry import landmarks_fusion


def test_fusion():
    # define test dataframes
    landmarks1 = pd.DataFrame(
        data=np.array([[2, 2, 2, 0.0], [3, 4, 3, 1.0], [4, 4, 4, 0.5]]),
        columns=["x", "y", "z", "visibility"],
    )

    landmarks2 = pd.DataFrame(
        data=np.array([[10, 10, 10, 1.0], [8, 8, 8, 0.0], [3, 3, 3, 0.5]]),
        columns=["x", "y", "z", "visibility"],
    )

    # apply function
    result = landmarks_fusion(world_coordinates=[landmarks1, landmarks2])

    # define true answer
    real_answer = pd.DataFrame(
        data=np.array([[10, 10, 10], [3, 4, 3], [3.5, 3.5, 3.5]]),
        columns=["x", "y", "z"],
    )

    # compare
    assert np.linalg.norm(result.values - real_answer.values) < 1e-3


test_fusion()
