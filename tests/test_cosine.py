import os
import sys

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from utils.geometry import cosine
import numpy as np
from math import cos, pi, sqrt


def test_cosine():
    # to check floats
    eps = 1e-6

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    assert abs(cosine(v1, v2)) < eps, "Incorrect value."

    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([1.0, 1.0, 1.0])

    assert abs(cosine(v1, v2) - 1.0 / sqrt(3.0)) < eps, "Incorrect value."


test_cosine()
