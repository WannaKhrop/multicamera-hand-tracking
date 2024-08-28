"""
Module contains some usefull functions for detection of hand landmarks and their processing

Author: Ivan Khrop
Date: 23.07.2024
"""
# import basic mediapipe components
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import NormalizedLandmark, Landmark

# import compenents for math and plots
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import cv2

# import additional staff
from camera_thread.camera import camera
from hand_recognition.HandLandmarks import HandLandmark
from hand_recognition.LandmarkerSingleton import LandmarkerSingleton
from hand_recognition.HolisticLandmarkerSingleton import HolisticLandmarkerSingleton

# RealSense for typing
import pyrealsense2 as rs

from utils.constants import CAMERA_RESOLUTION_HEIGHT, CAMERA_RESOLUTION_WIDTH
from utils.geometry import assign_visability


def process_image(
    image: np.ndarray, holistic: bool = False
) -> mp.tasks.vision.HolisticLandmarkerResult:  # type: ignore
    """
    Return results of the detection for an image.

    Parameters
    ----------
    image: np.ndarray
        Image that must be processed
    holistic: bool = False
        Which model to use. By default => HandModel.
        If True, use HolisticModel

    Returns
    -------
    detection_results: mediapipe.tasks.python.vision.hand_landmarker.HandLandmarkerResult
        Results of mediapipe
    """
    # convert image to mediapipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # process image
    if holistic:
        landmarker = HolisticLandmarkerSingleton()
        results = landmarker.process(image)
    else:
        landmarker = LandmarkerSingleton()
        results = landmarker.detect(mp_image)

    return results


def draw_hand(
    hand_landmarks: list[NormalizedLandmark],  # type: ignore
    azimuth: int = 10,
    elevation: int = 10,  # type: ignore
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
        Value to elevate the 3D-plot
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


def draw_hand_animated(
    hand_landmarks: list[tuple[int, pd.DataFrame]],
    azimuth: int = 10,
    elevation: int = 10,  # type: ignore
):
    """
    Draw 3D model for the provided list of landmarks.

    Parameters
    ----------
    pd.DataFrame
        List of detected handlandmarks to be drawn
    azimuth: int
        Angle to turn the 3D-plot
    elevation:
        Value to elevate the 3D-plot
    """

    def _normalize_color(color):
        return tuple(v / 255.0 for v in color)

    landmark_drawing_spec = solutions.drawing_styles.get_default_hand_landmarks_style()
    connection_drawing_spec = (
        solutions.drawing_styles.get_default_hand_connections_style()
    )

    # create initials
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)

    # Set the limits of the axes
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

    # Function to update the plot
    def update(frame_id):
        # clear plot
        ax.clear()
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)

        # get next set of landmarks
        landmarks = hand_landmarks[frame_id][1]  # get pd.DataFrame
        plotted_landmarks = {}

        for idx in landmarks.index:
            if landmarks.loc[idx].z < 1e-3:
                continue

            ax.scatter3D(
                xs=[landmarks.loc[idx].x],
                ys=[landmarks.loc[idx].y],
                zs=[landmarks.loc[idx].z],
                color=_normalize_color(landmark_drawing_spec[idx].color[::-1]),
                linewidth=landmark_drawing_spec[idx].thickness,
            )
            plotted_landmarks[idx] = (
                landmarks.loc[idx].x,
                landmarks.loc[idx].y,
                landmarks.loc[idx].z,
            )

            ax.text(
                landmarks.loc[idx].x,
                landmarks.loc[idx].y,
                landmarks.loc[idx].z,
                str(idx),
                size=12,
                zorder=0,
                color="black",
            )

        num_landmarks = len(landmarks)

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

            if landmarks.loc[start_idx].z < 1e-3 or landmarks.loc[end_idx].z < 1e-3:
                continue

            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]

                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec[key].color[::-1]),
                    linewidth=connection_drawing_spec[key].thickness,
                )

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=np.arange(0, len(hand_landmarks)), interval=200, blit=False
    )
    ani.save("3d_animation.gif", writer="pillow", fps=5)

    plt.show()


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
    new_coordinates -= detection_result[closest_landmark.value]

    return new_coordinates


def get_depth_data_from_pixel(
    x: float, y: float, depth_frame: np.ndarray, intrinsics: rs.pyrealsense2.intrinsics
) -> np.array:
    """
    Get camera coordinates for pixel (x, y).

    This method also assigns visibility to each landmark.
    Coordinates with small visibility are absolutely unreliable !!!

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
    assign_visability(landmarks)

    # for future
    relative_depth = landmarks["z"].copy()
    closest_point_landmark_idx = 0
    dist = 99.0

    # get coords
    coords = ["x", "y", "z"]
    for idx in landmarks.index:
        x, y = landmarks.loc[idx].x, landmarks.loc[idx].y
        landmarks.loc[idx, coords] = get_depth_data_from_pixel(
            x, y, depth_frame, intrinsics
        )

        # if distance to camera is small, then we did not recognize this point => visibility = 0.0
        if landmarks.loc[idx, "z"] < 1e-3:
            landmarks.loc[idx, "visibility"] = 0.0
        elif landmarks.loc[idx, "z"] < dist:
            dist = landmarks.loc[idx, "z"]
            closest_point_landmark_idx = idx

    # get the closest point to the camera according to z-axis
    closest_point_landmark = HandLandmark(closest_point_landmark_idx)

    # change origin and get relative depth_data
    relative_depth = (
        1.0 + relative_depth - relative_depth.values[closest_point_landmark.value]
    )
    # multiply with depth of the closest point
    real_depth = relative_depth * landmarks.loc[closest_point_landmark.value, "z"]
    landmarks["z"] = real_depth

    # handle zeros where depth was missed
    landmarks[landmarks["visibility"] < 1e-3] = np.zeros(4)

    return landmarks


def convert_to_camera_coordinates_holistic(
    mp_results: mp.tasks.vision.HolisticLandmarkerResult,  # type: ignore
    depth_frame: np.ndarray,
    intrinsics: rs.pyrealsense2.intrinsics,
) -> dict[str, pd.DataFrame]:
    """
    Apply mediapipe to color_frame and extract camera coordinates of each landmark.

    Parameters
    ----------
    color_frame: np.ndarray
        Picture for hand detection.
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


def convert_to_camera_coordinates(
    mp_results: mp.tasks.vision.HolisticLandmarkerResult,  # type: ignore
    depth_frame: np.ndarray,
    intrinsics: rs.pyrealsense2.intrinsics,
) -> dict[str, np.array]:
    """
    Apply mediapipe to color_frame and extract camera coordinates of each landmark.\
    
    Parameters
    ----------
    color_frame: np.ndarray
        Picture for hand detection.
    depth_frame: np.ndarray
        Depth data from image.
    intrinsics: rs.pyrealsense2.intrinsics
        Camera intrinsics parameters. 

    Returns
    -------
    dict[str, np.ndarray]
        Extracted landmarks for each hand.
    """
    hands = {}
    for idx, hand in enumerate(mp_results.handedness):
        # get index and hand
        name = hand[0].category_name

        # get hand world landmarks
        world_landmarks = to_numpy_ndarray(mp_results.hand_world_landmarks[idx])

        # get normalized landmarks
        landmarks = to_numpy_ndarray(mp_results.hand_landmarks[idx])

        # get the closest point to the camera according to z-axis
        closest_point_landmark = HandLandmark(np.argmin(world_landmarks[:, 2]))
        closest_point_idx = closest_point_landmark.value

        # identify pixels
        closest_point = get_depth_data_from_pixel(
            x=landmarks[closest_point_idx][0],
            y=landmarks[closest_point_idx][1],
            depth_frame=depth_frame,
            intrinsics=intrinsics,
        )

        # make the closest point a new center of coordinates
        hand_with_new_origin = change_origin(closest_point_landmark, world_landmarks)

        # add the real world coordinates to the camera coordinates
        # save result for hand
        hands[name] = closest_point + hand_with_new_origin

    return hands


def draw_landmarks_on_image(
    annotated_image: np.array, detection_result: list[NormalizedLandmark]
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

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


def draw_landmarks_holistics(
    annotated_image: np.array, detection_result: list[NormalizedLandmark]
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
