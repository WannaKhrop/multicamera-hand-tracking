"""
Module contains some usefull functions for detection of hand landmarks and their processing

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic mediapipe components
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import NormalizedLandmark, Landmark

# import compenents for math and plots
import numpy as np
import pandas as pd

# import additional staff
from camera_thread.camera import camera

# RealSense for typing
import pyrealsense2 as rs

from utils.constants import CAMERA_RESOLUTION_HEIGHT, CAMERA_RESOLUTION_WIDTH
from utils.geometry import assign_visibility
from utils.utils import TimeChecker


def to_numpy_ndarray(landmarks: list[NormalizedLandmark] | list[Landmark]):  # type: ignore
    """
    Convert list of Normalized landmarks into a numpy array.

    Parameters
    ----------
    landmarks: list[NormalizedLandmark] | list[Landmark]
        List of landmarks to be converted

    Returns
    -------
    np.ndarray:
        Resulting matrix [21, 3]
    """
    # get amount of landmarks
    num_landmarks = len(landmarks)

    matrix = np.zeros((num_landmarks, 3))

    for idx, landmark in enumerate(landmarks):
        matrix[idx] = np.array([landmark.x, landmark.y, landmark.z])

    return matrix


def to_numpy_ndarray_holistics(landmarks: landmark_pb2.NormalizedLandmarkList):
    """
    Convert list of Normalized landmarks into a numpy array.

    Parameters
    ----------
    landmarks: landmark_pb2.NormalizedLandmarkList
        List of landmarks to be converted

    Returns
    -------
    np.ndarray:
        Resulting matrix [21, 3]
    """
    matrix = list()

    for landmark in landmarks.landmark:
        matrix.append(np.array([landmark.x, landmark.y, landmark.z]))

    return np.array(matrix)


def hand_to_df(landmarks: np.ndarray):
    """
    Convert matrix of points into a pandas.DataFrame.
    Resulting DataFrame contains the following columns: index, x, y, z

    Parameters
    ----------
    landmarks: np.ndarray
        Landmarks to be converted

    Returns
    -------
    pd.DataFrame
    """
    columns = ["x", "y", "z"]
    df = pd.DataFrame(landmarks, columns=columns)
    return df


def get_depth_data_from_pixel(
    x: float, y: float, depth_frame: np.ndarray, intrinsics: rs.pyrealsense2.intrinsics
) -> np.ndarray:
    """
    Get camera coordinates for pixel (x, y).

    Parameters
    ----------
    x: float
        Normalized x-coordinate at the image.
    y: float
        Normalized y-coordinate at image.
    depth_frame: np.ndarray
        Depth data from image.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters.

    Returns
    -------
    np.array
        Camera cordinates.
    """
    # identify pixels
    x_pixel = int(x * CAMERA_RESOLUTION_WIDTH)
    y_pixel = int(y * CAMERA_RESOLUTION_HEIGHT)
    point = camera.get_camera_coordinates(x_pixel, y_pixel, depth_frame, intrinsics)

    return point


def get_data_from_pixel_depth(
    x: float, y: float, depth: float, intrinsics: rs.pyrealsense2.intrinsics
) -> np.ndarray:
    """
    Get camera coordinates for pixel (x, y) and depth.

    Parameters
    ----------
    x: float
        Normalized x-coordinate at the image.
    y: float
        Normalized y-coordinate at image.
    depth: float
        Depth data from image.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters.

    Returns
    -------
    np.array
        Camera cordinates.
    """
    # identify pixels
    x_pixel = int(x * CAMERA_RESOLUTION_WIDTH)
    y_pixel = int(y * CAMERA_RESOLUTION_HEIGHT)
    point = camera.get_coordinates_for_depth(x_pixel, y_pixel, depth, intrinsics)

    return point


def convert_hand_holistic(
    holistic_landmarks: landmark_pb2.NormalizedLandmarkList,
    depth_frame: np.ndarray,
    intrinsics: rs.pyrealsense2.intrinsics,
) -> pd.DataFrame:
    """
    Coverts results of holistic landmark-detection to camera coordinates.

    This method also assigns visibility to each landmark.
    Coordinates with small visibility are absolutely unreliable !!!

    Parameters
    ----------
    holistic_landmarks: landmark_pb2.NormalizedLandmarkList
        Landmarks of the hand.
    depth_frame: np.ndarray
        Depth data from image.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with camera coordinates and visibility like: [x, y, z, visibility]
    """
    # get normalized landmarks
    landmarks = hand_to_df(to_numpy_ndarray_holistics(holistic_landmarks))

    # visibility
    assign_visibility(landmarks)

    # for future
    closest_point_landmark_idx = 0
    min_dist = 1e2

    # !!! TODO !!!
    # rewrite it with precise world coordinates calculation !!!

    # get coordinates and identifz the closest point to the camera
    coords = ["x", "y", "z"]
    for idx in landmarks.index:
        x, y = landmarks.loc[idx].x, landmarks.loc[idx].y
        x_pixel = int(x * CAMERA_RESOLUTION_WIDTH)
        y_pixel = int(y * CAMERA_RESOLUTION_HEIGHT)
        depth = camera.get_depth(
            x_pixel=x_pixel, y_pixel=y_pixel, depth_frame=depth_frame
        )

        # if distance to camera is small, then we did not recognize this point
        if depth < min_dist and depth > 1e-3:
            min_dist, closest_point_landmark_idx = depth, idx

    # change origin and get relative depth_data
    landmarks.loc[:, "z"] = (
        1.0 + landmarks.loc[:, "z"] - landmarks.loc[closest_point_landmark_idx, "z"]
    )
    # multiply with depth of the closest point
    landmarks.loc[:, "z"] = min_dist * landmarks.loc[:, "z"]

    # handle zeros where depth was missed
    # now we have assumption about depth and use it to correct coordinates
    for idx in landmarks.index:
        x, y, depth = landmarks.loc[idx].x, landmarks.loc[idx].y, landmarks.loc[idx].z
        landmarks.loc[idx, coords] = get_data_from_pixel_depth(x, y, depth, intrinsics)

    return landmarks


@TimeChecker
def convert_to_camera_coordinates_holistic(
    mp_results: mp.tasks.vision.HolisticLandmarkerResult,  # type: ignore
    depth_frame: np.ndarray,
    intrinsics: rs.pyrealsense2.intrinsics,
) -> dict[str, pd.DataFrame]:
    """
    Apply mediapipe to color_frame and extract camera coordinates of each landmark.

    Parameters
    ----------
    mp_results: mp.tasks.vision.HolisticLandmarkerResult
        Results of detection.
    depth_frame: np.ndarray
        Depth data from image.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrame with camera coordinates and visibility like: [x, y, z, visibility] for each hand.
    """
    hands = dict()

    # for testing
    if mp_results.left_hand_landmarks is not None:
        # get result
        hands["Left"] = convert_hand_holistic(
            holistic_landmarks=mp_results.left_hand_landmarks,
            depth_frame=depth_frame,
            intrinsics=intrinsics,
        )

    if mp_results.right_hand_landmarks is not None:
        # get result
        hands["Right"] = convert_hand_holistic(
            holistic_landmarks=mp_results.right_hand_landmarks,
            depth_frame=depth_frame,
            intrinsics=intrinsics,
        )

    return hands


def draw_landmarks_holistics(
    annotated_image: np.ndarray,
    detection_result: list[NormalizedLandmark],  # type: ignore
):  # type: ignore
    """
    Annotate image with detected landmarks.

    Parameters
    ----------
    annotated_image: np.array
        Image to annotate.
    detection_result: list[NormalizedLandmark]
        Landmarks to draw at the image.
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    if detection_result is not None:
        mp_drawing.draw_landmarks(
            annotated_image,
            detection_result,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
