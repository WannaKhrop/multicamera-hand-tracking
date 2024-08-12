"""
Module contains some geometrical functions that are used 
to calculate occlusions in 3D-space

Author: Ivan Khrop
Date: 27.07.2024
"""
import numpy as np
import pandas as pd
from typing import Callable
from shapely.geometry import Polygon

from utils.utils import softmax

from hand_recognition.HandLandmarks import (
    finger_connections,
    palm_landmarks,
    fingers_landmarks,
    HandLandmark,
)


def define_line(point1: np.ndarray, point2: np.ndarray) -> Callable:
    """
    Defines a line in 3D space given two points.
    point1, point2: numpy arrays that represent points in 3D space.
    Returns a function representing the parametric form of the line.

    Parameters
    ----------
    P1: np.ndarray
        First point of the line
    P2: np.ndarray
        Second point of the line

    Returns:
    Callable[float]:
        Function that gives a point of the line for parameter.
    """

    def line(t):
        return point1 + t * (point2 - point1)

    return line


def project_point_to_line(
    point1: np.ndarray, point2: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Projects a target onto a line defined by point1 and point2 in 3D space.
    point1, point2, target: numpy arrays representing points in 3D space.
    Returns the projection point P.

    Parameters
    ----------
    point1: np.ndarray
        First point of the line
    point2: np.ndarray
        Second point of the line
    target: np.ndarray
        Point that must be projected

    Returns:
    np.ndarray:
        Projection point of Q to the line P1 + t * (P2 - P1).
    """
    line_vector = point2 - point1  # Direction vector
    vector_to_point = target - point1  # Vector from P1 to target
    scale = np.dot(vector_to_point, line_vector) / np.dot(
        line_vector, line_vector
    )  # Projection scalar
    return point1 + scale * line_vector  # Projection point


def find_palm_plane(df_palm_landmarks: pd.DataFrame) -> np.ndarray:
    """
    Construct a palm plane.

    Parameters
    ----------
    df_palm_landmarks: pd.DataFrame
        Landmarks of mediapipe that define palm points = [0, 5, 9, 13, 17]

    Returns
    -------
    np.ndarray:
        A, B, C, D coefficients of a plne as numpy array
    """
    # get poits of interest
    index_finger_point = np.array(
        df_palm_landmarks.loc[HandLandmark.INDEX_FINGER_MCP.value]
    )
    pinky_finger_point = np.array(df_palm_landmarks.loc[HandLandmark.PINKY_MCP.value])
    wrist_point = np.array(df_palm_landmarks.loc[HandLandmark.WRIST.value])

    # construct a plane
    vector1 = index_finger_point - wrist_point
    vector2 = pinky_finger_point - wrist_point

    # get coefficients
    normal = np.cross(vector1, vector2)
    A, B, C = normal
    D = -np.dot(normal, wrist_point)

    return np.ndarray([A, B, C, D])


def project_point_to_plane(plane: np.ndarray, point: np.array) -> np.ndarray:
    """
    Project a point on a plane in 3D space.

    Parameters
    ----------
    plane: np.ndarray
        Plane as array [A, B, C, D]
    point: np.array
        Point to be projected

    Returns
    -------
    np.ndarray:
        Projection point
    """
    point_ext = np.hstac([point, np.ones(1)])
    normal_vector = np.array([plane[0], plane[1], plane[2]])

    d = np.dot(point_ext, plane) / np.linalg.norm(normal_vector)

    return point + d * normal_vector


def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine between two vectors.

    Parameters
    ----------
    v1: np.ndarray
        The first vector
    v2: np.ndarray
        The second vector

    Returns
    -------
    float:
        Cosine between two vectors
    """
    cosine = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return cosine


def is_between(point1: np.ndarray, point2: np.ndarray, target: np.ndarray):
    """
    Check if target point is located between point1 and point2.

    Parameters
    ----------
    point1: np.ndarray
        First point of the line
    point2: np.ndarray
        Second point of the line
    target: np.ndarray
        Point that must be projected

    Returns:
    bool:
        Returns True is target is located between two points
        Otherwise, returns False
    """
    # check conditions for two possible cases
    cond1 = point1[0] <= target[0] and target[0] <= point2[0]
    cond2 = point1[1] <= target[1] and target[1] <= point2[1]
    cond3 = point1[2] <= target[2] and target[2] <= point2[2]

    cond1_rev = point2[0] <= target[0] and target[0] <= point1[0]
    cond2_rev = point2[1] <= target[1] and target[1] <= point1[1]
    cond3_rev = point2[2] <= target[2] and target[2] <= point1[2]

    result = (cond1 and cond2 and cond3) or (cond1_rev and cond2_rev and cond3_rev)

    return result


def construct_palm_polygon(df_palm_landmarks: pd.DataFrame) -> Polygon:
    """
    Construct a polygon for the landmarks of a palm.

    Parameters
    ----------
    df_palm_landmarks: pd.DataFrame
        Landmarks of mediapipe that define palm points = [0, 5, 9, 13, 17]

    Returns
    -------
    Polygon:
        Polygon of points that define a palm
    """
    # firstly let's find bottom points of the palm
    PALM_INDEX_FINGER = HandLandmark.INDEX_FINGER_MCP.value
    PALM_MIDDLE_FINGER = HandLandmark.MIDDLE_FINGER_MCP.value
    PALM_RING_FINGER = HandLandmark.RING_FINGER_MCP.value
    PALM_PINKY_FINGER = HandLandmark.PINKY_MCP.value
    PALM_WRIST = HandLandmark.WRIST.value

    # get projection vector of wrist to index finger - small finger line
    projection_point = project_point_to_line(
        point1=np.array(df_palm_landmarks.loc[PALM_INDEX_FINGER]),
        point2=np.array(df_palm_landmarks.loc[PALM_PINKY_FINGER]),
        target=np.array(df_palm_landmarks.loc[PALM_WRIST]),
    )

    projection_vector = np.array(df_palm_landmarks.loc[PALM_WRIST]) - projection_point

    # identify additional points of palm near the wrist
    wrist_left = np.array(df_palm_landmarks.loc[PALM_INDEX_FINGER]) + projection_vector
    wrist_right = np.array(df_palm_landmarks.loc[PALM_PINKY_FINGER]) + projection_vector

    polygon = Polygon(
        [
            np.array(df_palm_landmarks.loc[PALM_INDEX_FINGER]),
            np.array(df_palm_landmarks.loc[PALM_MIDDLE_FINGER]),
            np.array(df_palm_landmarks.loc[PALM_RING_FINGER]),
            np.array(df_palm_landmarks.loc[PALM_PINKY_FINGER]),
            wrist_right,
            np.array(df_palm_landmarks.loc[PALM_WRIST]),
            wrist_left,
        ]
    )

    return polygon


def assign_visability(df_landmarks: pd.DataFrame):
    """
    Assign visability level to each finger landmark.
    This function adds an additional column "visibility" to the DataFrame

    Parameters
    ----------
    df_landmarks: pd.DataFrame
        Landmarks in DataFrame format with columns: index, x, y, z
    """
    # sort landmarks by z coordinate
    df_landmarks_sorted = df_landmarks.sort_values(by="z", ascending=True)

    # camera vector
    camera_vector = np.array([0.0, 0.0, -1.0])

    # create polygon using palm landmarks
    palm_poligon = construct_palm_polygon(df_landmarks.loc[palm_landmarks])
    palm_plane = find_palm_plane(df_landmarks.loc[palm_landmarks])

    visibility_dict = dict()
    # go over all landmarks
    for idx in df_landmarks_sorted.index:
        # if it's a palm landmark then assign visibility = 1.0
        if idx not in fingers_landmarks:
            visibility_dict[idx] = 1.0
            continue

        # at the beginning we assume maximal visibility
        visibility = 1.0

        # go over all hand connections and check if this landmarks is hiiden by another finger
        for p_idx_1, p_idx_2 in finger_connections:
            # if this landmark is a part of the segment => no need to check
            if p_idx_1 == idx or p_idx_2 == idx:
                continue

            # if at least one point of the segment is further from camera => no need to check
            if (
                df_landmarks_sorted.loc[p_idx_1].z >= df_landmarks_sorted.loc[idx].z
            ) or (df_landmarks_sorted.loc[p_idx_2].z >= df_landmarks_sorted.loc[idx].z):
                continue

            projection_point = project_point_to_line(
                point1=np.array(df_landmarks_sorted.loc[p_idx_1]),
                point2=np.array(df_landmarks_sorted.loc[p_idx_2]),
                target=np.array(df_landmarks_sorted.loc[idx]),
            )

            # check if projection is actually clamped by two other landmarks
            if is_between(
                point1=np.array(df_landmarks_sorted.loc[p_idx_1]),
                point2=np.array(df_landmarks_sorted.loc[p_idx_2]),
                target=projection_point,
            ):
                projection_vector = projection_point - np.array(
                    df_landmarks_sorted.loc[idx]
                )
                cosine_value = cosine(projection_vector, camera_vector)

                visibility = min(visibility, min(1.0, 1.0 - cosine_value))

        # check if a landmark is actually hidden by the palm
        palm_projection = project_point_to_plane(
            palm_plane, np.array(df_landmarks_sorted.loc[idx])
        )

        # if projection point is inside of palm
        if palm_poligon.contains(palm_projection):
            # get a projections vector
            palm_projection_vector = palm_projection - np.array(
                df_landmarks_sorted.loc[idx]
            )

            cosine_value = cosine(palm_projection_vector, camera_vector)

            visibility = min(visibility_dict, min(1.0, 1.0 - cosine_value))

        visibility_dict[idx] = visibility

    # store results
    df_landmarks["visibility"] = df_landmarks.index.map(visibility_dict)

    # sort by index (HandLandmark id)
    df_landmarks = df_landmarks.sort_index()


def landmarks_fusion(
    world_coordinates: list[pd.DataFrame], softmax_const: float = 20.0
) -> pd.DataFrame:
    """
    Combine information from different cameras and get the resulting set of landmarks.

    Parameters
    ----------
    world_coordinates: list[pd.DataFrame]
        World coordinates with assignment of visibility for each landmark.
        Each DataFrame contains 21 row with columns = [x, y, z, visibility]
    softmax_const: float = 20.0
        Constant that is used for fusion to highlight large visibility.

    Returns
    -------
    pd.DataFrame
        The resulting set of landmarks.
    """
    # define name of column
    x, y, z, vis = "x", "y", "z", "visibility"

    # get all the visibilities and matrixes
    visibilities = np.hstack(
        [frame[vis].values.reshape(-1, 1) for frame in world_coordinates]
    )
    matrixes = np.array([frame[[x, y, z]].values for frame in world_coordinates])

    # apply softmax and split results
    weights = softmax(data=visibilities, temperature=softmax_const)
    weights = np.array(np.hsplit(weights, weights.shape[1]))

    # final result
    result = np.sum(matrixes * weights, axis=0)

    return pd.DataFrame(data=result, columns=[x, y, z])
