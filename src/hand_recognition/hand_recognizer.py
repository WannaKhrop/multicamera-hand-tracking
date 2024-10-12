"""
Module contains some usefull functions for detection of hand landmarks and their processing

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic mediapipe components
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import NormalizedLandmark

# import compenents for math and plots
import numpy as np
import pandas as pd

# import additional staff
from camera_thread.camera import camera

# RealSense for typing
import pyrealsense2 as rs

from utils.constants import CAMERA_RESOLUTION_HEIGHT, CAMERA_RESOLUTION_WIDTH
from utils.geometry import assign_visibility


def to_numpy_ndarray(landmarks: landmark_pb2.NormalizedLandmarkList):
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


def convert_hand(
    holistic_landmarks: landmark_pb2.NormalizedLandmarkList,
) -> pd.DataFrame:
    """
    Coverts results of holistic landmark-detection to pandas dataframe and assign visibility.

    Parameters
    ----------
    holistic_landmarks: landmark_pb2.NormalizedLandmarkList
        Landmarks of the hand.

    Returns
    -------
    pd.DataFrame
        DataFrame with camera coordinates and visibility like: [x, y, z, visibility]
    """
    # get normalized landmarks
    landmarks = hand_to_df(to_numpy_ndarray(holistic_landmarks))

    # visibility
    assign_visibility(landmarks)

    return landmarks


def convert_to_features(landmarks: pd.DataFrame, depth_frame: np.ndarray) -> np.ndarray:
    """
    Convert landmarks to features using depth frame.

    Parameters
    ----------
    landmarks: pd.DataFrame
        DataFrame with camera coordinates and visibility like: [x, y, z, visibility.
    depth_frame: np.ndarray
        Depth data from image.

    Returns
    -------
    np.ndarray
        Features as vector.
    """
    # get coordinates and identifz the closest point to the camera
    depths: list[float] = list()
    x = (landmarks.x.values * CAMERA_RESOLUTION_WIDTH).astype(int)
    y = (landmarks.y.values * CAMERA_RESOLUTION_HEIGHT).astype(int)
    for x_pixel, y_pixel in zip(x, y):
        # get depths and save
        depth = camera.get_depth(
            x_pixel=x_pixel, y_pixel=y_pixel, depth_frame=depth_frame
        )
        depths.append(depth)

    # constuct features
    return np.hstack([landmarks.z.values, np.array(depths)])


def retrieve_from_depths(
    landmarks: pd.DataFrame, depths: np.ndarray, intrinsics: rs.pyrealsense2.intrinsics
):
    """
    Retrieve camera coordinates for landmarks having depths.

    Parameters
    ----------
    holistic_landmarks: landmark_pb2.NormalizedLandmarkList
        Landmarks of the hand. Will be changed !!!
    depths: np.ndarray
        Predicted depths.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters.
    """
    coords = ["x", "y", "z"]
    # now we have assumption about depth and use it to correct coordinates
    for idx, depth in zip(landmarks.index, depths, strict=True):
        x, y = landmarks.loc[idx].x, landmarks.loc[idx].y
        landmarks.loc[idx, coords] = get_data_from_pixel_depth(x, y, depth, intrinsics)


def extract_landmarks(
    mp_results: mp.tasks.vision.HolisticLandmarkerResult,  # type: ignore
) -> dict[str, pd.DataFrame]:
    """
    Extract each landmark.

    Parameters
    ----------
    mp_results: mp.tasks.vision.HolisticLandmarkerResult
        Results of detection.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrame with camera coordinates and visibility like: [x, y, z, visibility] for each hand.
    """
    hands = dict()

    # for testing
    if mp_results.left_hand_landmarks is not None:
        # get result
        hands["Left"] = convert_hand(
            holistic_landmarks=mp_results.left_hand_landmarks,
        )

    if mp_results.right_hand_landmarks is not None:
        # get result
        hands["Right"] = convert_hand(
            holistic_landmarks=mp_results.right_hand_landmarks
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
