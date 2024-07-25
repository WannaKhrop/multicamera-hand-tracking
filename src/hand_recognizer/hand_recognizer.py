"""
Module contains some usefull functions for detection of hand landmarks and their processing

Author: Ivan Khrop
Date: 23.07.2024
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.components.containers import NormalizedLandmark, Landmark

from matplotlib import pyplot as plt
import numpy as np

from HandLandmarker import HandLandmark

# define types
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


def create_landmarker(path_to_model: str) -> HandLandmarker:
    """
    Create a new instance of HandLandmarker.

    Parameters
    ----------
    path_to_model: str
        Path to file *.task for mediapipe-hands

    Returns
    -------
    HandLandmarker:
        Landmarker that is ready for use
    """
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path_to_model),
        num_hands=2,
        running_mode=VisionRunningMode.IMAGE,
    )

    return HandLandmarker.create_from_options(options)


def process_image(
    landmarker: HandLandmarker, image: np.ndarray
) -> mp.tasks.vision.HandLandmarkerResult:
    """
    Return results of the detection for an image.

    Parameters
    ----------
    landmarker: HandLandmarker
        Landmarker to detect hands and landmarks on an image
    image: np.ndarray
        Image that must be processed

    Returns
    -------
    detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
        Results of mediapipe
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    return landmarker.detect(mp_image)


def draw_hand(
    hand_landmarks: list[NormalizedLandmark], azimuth: int = 10, elevation: int = 10
):
    """
    Draw 3D model for the provided list of landmarks.

    Parameters
    ----------
    hand_landmarks: list[NormalizedLandmark]
        List of detected handlandmarks to be drawn
    azimuth: int
        Angle to turn the 3D-plot
    elevation:
        Vakue to elevate the 3D-plot
    """

    def _normalize_color(color):
        return tuple(v / 255.0 for v in color)

    landmark_drawing_spec = solutions.drawing_styles.get_default_hand_landmarks_style()
    connection_drawing_spec = (
        solutions.drawing_styles.get_default_hand_connections_style()
    )

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}

    for idx, landmark in enumerate(hand_landmarks):
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec[idx].color[::-1]),
            linewidth=landmark_drawing_spec[idx].thickness,
        )
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        ax.text(
            -landmark.z,
            landmark.x,
            -landmark.y,
            str(idx),
            size=12,
            zorder=0,
            color="black",
        )

    num_landmarks = len(hand_landmarks)

    # Draws the connections if the start and end landmarks are both visible.
    for connection in solutions.hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        key = (start_idx, end_idx)
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(
                f"Landmark index is out of range. Invalid connection "
                f"from landmark #{start_idx} to landmark #{end_idx}."
            )

        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [plotted_landmarks[start_idx], plotted_landmarks[end_idx]]

            ax.plot3D(
                xs=[landmark_pair[0][0], landmark_pair[1][0]],
                ys=[landmark_pair[0][1], landmark_pair[1][1]],
                zs=[landmark_pair[0][2], landmark_pair[1][2]],
                color=_normalize_color(connection_drawing_spec[key].color[::-1]),
                linewidth=connection_drawing_spec[key].thickness,
            )
    plt.show()


def to_numpy_ndarray(landmarks: list[NormalizedLandmark] | list[Landmark]):
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


def change_origin(
    closest_landmark: HandLandmark, detection_result: np.ndarray
) -> np.ndarray:
    """
    Construct model of a hand using one landmark and hand_world_coordinates of detected points.

    Parameters
    ----------
    closest_landmark: HandLandmark
        New origin for all the detected points
    detection_results: list[NormalizedLandmark]
        Result of mediapipe hand detection

    Returns
    -------
    np.ndarray:
        A matrix [21, 3] where closest_ladmark is an origin with respect to other landmarks
    """
    # copy the original matrix of coordinates and substract a new origin
    new_coordinates = np.copy(detection_result)
    new_coordinates -= detection_result[closest_landmark]

    return new_coordinates
