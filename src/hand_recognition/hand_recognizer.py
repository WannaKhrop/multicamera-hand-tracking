"""
Module contains some usefull functions for detection of hand landmarks and their processing

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic mediapipe components
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.components.containers import NormalizedLandmark, Landmark

# import compenents for math and plots
from matplotlib import pyplot as plt
import numpy as np

# import additional staff
from camera_thread.camera import camera
from hand_recognition.HandLandmarks import HandLandmark
from hand_recognition.Landmarker import Landmarker


# picture constants
CAMERA_RESOLUTION_WIDTH = 1920
CAMERA_RESOLUTION_HEIGHT = 1080


def process_image(image: np.ndarray) -> mp.tasks.vision.HandLandmarkerResult:
    """
    Return results of the detection for an image.

    Parameters
    ----------
    image: np.ndarray
        Image that must be processed

    Returns
    -------
    detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
        Results of mediapipe
    """
    landmarker = Landmarker()
    # convert image to mediapipe
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


def convert_to_camera_coordinates(
    color_frame: np.ndarray, depth_frame, intrinsics
) -> dict[str, np.ndarray]:
    """
    Apply mediapipe to color_frame and extract camera coordinates of each landmark.\
    
    Parameters
    ----------
    color_frame: np.ndarray
        Picture for hand detection
    depth_frame: !!! TODO !!!
    intrinsics: !!! TODO !!!

    Returns
    -------
    dict[str, pd.DataFrame]
        Extracted landmarks for each hand.
    """
    results_of_mediapipe = process_image(color_frame)
    hands = {}
    for hand in results_of_mediapipe.handedness:
        # get index and hand
        idx = hand[0].index
        name = hand[0].category_name

        # get hand world landmarks
        world_landmarks = to_numpy_ndarray(
            results_of_mediapipe.hand_world_landmarks[idx]
        )

        # get normalized landmarks
        landmarks = to_numpy_ndarray(results_of_mediapipe.hand_landmarks[idx])

        # get the closest point to the camera according to z-axis
        closest_point_idx = HandLandmark(np.argmin(landmarks[:, 2])).value

        # identify pixels
        x_pixel = landmarks[closest_point_idx][0] * CAMERA_RESOLUTION_WIDTH
        y_pixel = landmarks[closest_point_idx][1] * CAMERA_RESOLUTION_HEIGHT
        closest_point = camera.get_depth_data_from_history(
            x_pixel, y_pixel, depth_frame, intrinsics
        )

        # make the closest point a new center of coordinates
        hand_with_new_origin = change_origin(closest_point_idx, world_landmarks)

        # add the real world coordinates to the camera coordinates
        # save result for hand
        hands[name] = closest_point + hand_with_new_origin
