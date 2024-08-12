import os
import sys

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from hand_recognition.hand_recognizer import change_origin, HandLandmark
import numpy as np


def test_origin_change():
    # create an initial array
    detected_coordinates = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
    # resulting array
    answer = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    # apply function
    chaneg_arr = change_origin(
        closest_landmark=HandLandmark(2), detection_result=detected_coordinates
    )

    # check
    assert np.linalg.norm(chaneg_arr - answer) < 1e-3, "Incorrect result"
