"""
Module contains some geometrical functions that are used 
to calculate occlusions in 3D-space

Author: Ivan Khrop
Date: 27.07.2024
"""
import numpy as np
import pandas as pd

from utils.utils import softmax, TimeChecker
from matplotlib import pyplot as plt

from hand_recognition.HandLandmarks import (
    finger_connections,
    palm_landmarks,
    HandLandmark,
)


def draw_palm_polygon(polygon: list[np.array], point: np.array):
    """
    Draw palm polygon and point of projection.

    For testing purposes.

    Parameters
    ----------
    polygon: list[np.array]
        Polygon with points.
    point: np.array
        Point to draw.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    np_polygon = np.array(polygon)
    # Plot the line
    ax.plot(
        np_polygon[:, 0],
        np_polygon[:, 1],
        np_polygon[:, 2],
        color="blue",
        label="3D Line",
    )

    # Plot the points
    ax.scatter(
        np_polygon[:, 0],
        np_polygon[:, 1],
        np_polygon[:, 2],
        color="red",
        s=100,
        label="Palm",
    )
    ax.scatter(point[0], point[1], point[2], color="green", s=100, label="Point")

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


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
    # Direction vector
    line_vector = point2 - point1
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)

    # Vector from P1 to target
    vector_to_point = target - point1

    # projection length
    scale = np.dot(vector_to_point, line_vector_normalized)

    return point1 + scale * line_vector_normalized  # Projection point


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
    D = -1.0 * np.dot(normal, wrist_point)

    return np.array([A, B, C, D], dtype=float)


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
    # get points and normal vector
    point_ext = np.hstack([point, np.ones(1)])
    normal_vector = np.array([plane[0], plane[1], plane[2]])
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    d = np.dot(point_ext, plane) / np.linalg.norm(normal_vector)

    return point - d * normal_vector_normalized


def is_inside_palm(polygon: list[np.array], plane: np.array, point: np.array) -> bool:
    """
    Check if point is inside a poligon. All points of polygon belong to one plane !!!

    Paramerters
    -----------
    polygon: list[np.array]
        Polygon of points that defines a convex hull palm.
    plane: np.array
        Plane that defines polygon.
    point: np.array
        Point to be checked.

    Returns
    -------
        True, if point belongs to polygon.
        False, otherwise.
    """
    # results
    list_of_checks = []
    # get normal vector of the plane
    normal_vector = plane[:3] / np.linalg.norm(plane[:3])  # x, y, z - coordinates

    for i in range(1, len(polygon)):
        p_start, p_finish = polygon[i - 1], polygon[i]

        # create vectors
        vect_start = point - p_start
        vect_finish = p_finish - point

        # get normalized dot product
        cross_product = np.cross(vect_start, vect_finish)
        cross_product /= np.linalg.norm(cross_product)

        # get cosine with plane normal vector (can be 1 or -1)
        list_of_checks.append(np.dot(cross_product, normal_vector))

    # if all the time we have the same oriented cross-product
    result = np.array(list_of_checks)

    return (result > 0).all() or (result < 0).all()


def construct_palm_polygon(
    df_palm_landmarks: pd.DataFrame, plane: np.array
) -> list[np.array]:
    """
    Construct a polygon for the landmarks of a palm.

    Parameters
    ----------
    df_palm_landmarks: pd.DataFrame
        Landmarks of mediapipe that define palm points = [0, 5, 9, 13, 17].
    plane: np.array
        Plane [A, B, C, D] that defines plane of a palm.

    Returns
    -------
    list[np.array]
        Polygon of points that defines palm like:
        INDEX=>MIDDLE=>RING=>PINKY=>PALM_BOTTOM_RIGHT=>WRIST=>PALM_BOTTOM_LEFT=>INDEX.
        All points belong to palm plane.
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

    # project middle finger on plane
    middle_finger_projection = project_point_to_plane(
        plane=plane, point=np.array(df_palm_landmarks.loc[PALM_MIDDLE_FINGER])
    )

    # ring finger projection
    ring_finger_projection = project_point_to_plane(
        plane=plane, point=np.array(df_palm_landmarks.loc[PALM_RING_FINGER])
    )

    polygon = [
        np.array(df_palm_landmarks.loc[PALM_INDEX_FINGER]),
        middle_finger_projection,
        ring_finger_projection,
        np.array(df_palm_landmarks.loc[PALM_PINKY_FINGER]),
        wrist_right,
        np.array(df_palm_landmarks.loc[PALM_WRIST]),
        wrist_left,
        np.array(df_palm_landmarks.loc[PALM_INDEX_FINGER]),
    ]

    return polygon


@TimeChecker
def assign_visibility(df_landmarks: pd.DataFrame):
    """
    Assign visability level to each finger landmark.
    This function adds an additional column "visibility" to the DataFrame

    Parameters
    ----------
    df_landmarks: pd.DataFrame
        Landmarks in DataFrame format with columns: index, x, y, z
    """
    # Camera vector
    camera_vector = np.array([0.0, 0.0, -1.0]).reshape(-1, 1)

    # Create polygon using palm landmarks
    palm_plane = find_palm_plane(df_landmarks.loc[palm_landmarks])
    palm_polygon = construct_palm_polygon(df_landmarks.loc[palm_landmarks], palm_plane)

    # Coordinates of all points
    coords = ["x", "y", "z"]
    all_points = df_landmarks[coords].values.T  # 3x21

    # Initialize visibility array
    visibility = np.ones(len(df_landmarks))  # 21,

    # Vectorized line projection for finger connections
    for p_idx_1, p_idx_2 in finger_connections:
        point1 = df_landmarks.loc[p_idx_1, coords].values.reshape((-1, 1))  # 3x1
        point2 = df_landmarks.loc[p_idx_2, coords].values.reshape((-1, 1))  # 3x1

        # Calculate projections for all points
        line_vector = point2 - point1  # 3x1 - 3x1 = 3x1
        line_vector_normalized = line_vector / np.linalg.norm(
            line_vector
        )  # 3x1 / const = 3x1
        vectors_to_points = all_points - point1  # 3x21 - 3x1 = 3x21 // broadcast
        scales = np.dot(
            vectors_to_points.T, line_vector_normalized
        )  # 21x3 dot 3x1 = 21x1
        projection_points = point1 + np.dot(
            line_vector_normalized, scales.T
        )  # 3x1 + 3x1 dot 1x21 = 3x21

        # Check if projections are between points (clamping)
        mask_between = is_between(point1, point2, projection_points)  # 21,

        # Calculate projection vectors and process self projections
        projection_vectors = projection_points - all_points  # 3x21 - 3x21 = 3x21
        projection_vectors_norms = np.linalg.norm(
            projection_vectors, axis=0, keepdims=True
        )  # 1x21
        projection_vectors_norms = np.where(
            projection_vectors_norms > 1e-3, projection_vectors_norms, 1.0
        )  # handle zero vectors to avoi division
        projection_vectors /= projection_vectors_norms  # 3x21

        cosines = np.squeeze(
            np.dot(projection_vectors.T, camera_vector)
        )  # 21x3 dot 3x1 = squeeze(21x1) = 21,

        # Update visibility where projections are between
        visibility[mask_between] = np.minimum(
            visibility[mask_between], 1.0 - cosines[mask_between]
        )

    # Check palm occlusion for points not in palm landmarks
    not_palm_mask = np.isin(df_landmarks.index, palm_landmarks, invert=True)  # 21,
    palm_projections = np.array(
        [
            project_point_to_plane(palm_plane, landmark)
            for landmark in all_points.T[not_palm_mask]
        ]
    )  # 21x3

    inside_palm_mask = np.array(
        [
            is_inside_palm(palm_polygon, palm_plane, projection)
            for projection in palm_projections
        ],
        dtype=bool,
    )  # 21,
    palm_projection_vectors = (
        palm_projections[inside_palm_mask]
        - all_points.T[not_palm_mask][inside_palm_mask]
    )  # 21x3

    cosines_palm = np.squeeze(
        np.dot(palm_projection_vectors, camera_vector)
    )  # 21x3 dot 3x1 = squeeze(21x1) = 21,

    # Update visibility for palm occlusion
    visibility[not_palm_mask][inside_palm_mask] = np.minimum(
        visibility[not_palm_mask][inside_palm_mask], 1.0 - cosines_palm
    )

    # Store results
    df_landmarks["visibility"] = visibility


def is_between(point1, point2, target):
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
    # Check for each dimension if the points are between the two landmarks
    cond1 = np.logical_and(point1[0, 0] <= target[0], target[0] <= point2[0, 0])
    cond2 = np.logical_and(point1[1, 0] <= target[1], target[1] <= point2[1, 0])
    cond3 = np.logical_and(point1[2, 0] <= target[2], target[2] <= point2[2, 0])

    cond1_rev = np.logical_and(point2[0, 0] <= target[0], target[0] <= point1[0, 0])
    cond2_rev = np.logical_and(point2[1, 0] <= target[1], target[1] <= point1[1, 0])
    cond3_rev = np.logical_and(point2[2, 0] <= target[2], target[2] <= point1[2, 0])

    result = np.logical_or(
        np.logical_and(cond1, np.logical_and(cond2, cond3)),
        np.logical_and(cond1_rev, np.logical_and(cond2_rev, cond3_rev)),
    )

    return result


def landmarks_fusion(
    world_coordinates: list[pd.DataFrame], softmax_const: float = 50.0
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
        [frame.loc[:, vis].values.reshape(-1, 1) for frame in world_coordinates]
    )
    matrixes = np.array([frame.loc[:, [x, y, z]].values for frame in world_coordinates])

    # apply softmax and split results
    weights = softmax(data=visibilities, temperature=softmax_const)
    weights = np.array(np.hsplit(weights, weights.shape[1]))

    # final result
    result = np.sum(matrixes * weights, axis=0)

    return pd.DataFrame(data=result, columns=[x, y, z])
