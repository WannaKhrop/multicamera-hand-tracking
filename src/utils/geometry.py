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


def draw_palm_polygon(polygon: list[np.ndarray], point: np.ndarray):
    """
    Draw palm polygon and point of projection.

    For testing purposes.

    Parameters
    ----------
    polygon: list[np.ndarray]
        Polygon with points.
    point: np.ndarray
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
        First point of the line.
    point2: np.ndarray
        Second point of the line.
    target: np.ndarray
        Point that must be projected.

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
        Landmarks of mediapipe that define palm points = [0, 5, 9, 13, 17].

    Returns
    -------
    np.ndarray:
        A, B, C, D coefficients of a palne as np.ndarray.
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


def project_point_to_plane(plane: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Project a point on a plane in 3D space.

    Parameters
    ----------
    plane: np.ndarray
        Plane as np.ndarray [A, B, C, D].
    points: np.ndarray
        Points to be projected as [Nx3].

    Returns
    -------
    np.ndarray:
        Projection point.
    """
    # get points and normal vector
    points_ext = np.hstack([points, np.ones(shape=(points.shape[0], 1))])
    normal_vector = np.array([plane[0], plane[1], plane[2]])
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    # find projections on plane for all landmarks
    distances = np.dot(points_ext, plane.reshape(-1, 1)) / np.linalg.norm(normal_vector)

    return points - distances * normal_vector_normalized


def is_inside_palm(
    polygon: np.ndarray, plane: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Check if points are inside a poligon. All points of polygon belong to one plane.

    Paramerters
    -----------
    polygon: np.array
        Polygon of points that defines a convex hull palm as [N_POLYGONx3].
    plane: np.array
        Plane that defines polygon.
    points: np.array
        Points to be checked as [Nx3].

    Returns
    -------
    np.ndarray
        True, if point belongs to polygon.
        False, otherwise.
    """
    # get normal vector of the plane
    normal_vector = plane[:3] / np.linalg.norm(plane[:3])  # x, y, z - coordinates

    p_start = np.array(polygon).T  # 3xN_POLYVERT
    p_finish = np.roll(p_start, shift=-1, axis=1)  # 3xN_POLYVERT

    # vectors calculations using broadcasting
    vect_start = (
        points[:, :, np.newaxis] - p_start[np.newaxis, :, :]
    )  # N_POINTx3x1 - 1x3xN_POLYVERT = N_POINTx3xN_POLYVERT
    vect_finish = (
        p_finish[np.newaxis, :, :] - points[:, :, np.newaxis]
    )  # N_POINTx3x1 - 1x3xN_POLYVERT = N_POINTx3xN_POLYVERT

    # get cross product vectors
    cross_product = np.cross(
        vect_start, vect_finish, axis=1
    )  # N_POINTx3xN_POLYVERT cross N_POINTx3xN_POLYVERT = N_POINTx3xN_POLYVERT

    # normalize and take care of vectors with zero norms
    cross_product_norms = np.linalg.norm(cross_product, axis=1, keepdims=True)
    cross_product_norms = np.where(
        cross_product_norms <= 1e-3, 1.0, cross_product_norms
    )
    cross_product /= cross_product_norms

    # dot product with plane normal
    dot_product = np.tensordot(
        cross_product, normal_vector, axes=([1], [0])
    )  # N_POINTx3xN_POLYVERT tensordot 3x1 = N_POINTxN_POLYVERT

    # final decision as bool vector
    positive_results = (dot_product >= 0.0).all(axis=1)
    negative_results = (dot_product <= 0.0).all(axis=1)

    return positive_results | negative_results


def construct_palm_polygon(
    df_palm_landmarks: pd.DataFrame, plane: np.ndarray
) -> list[np.ndarray]:
    """
    Construct a polygon for the landmarks of a palm.

    Parameters
    ----------
    df_palm_landmarks: pd.DataFrame
        Landmarks of mediapipe that define palm points = [0, 5, 9, 13, 17].
    plane: np.ndarray
        Plane [A, B, C, D] that defines plane of a palm.

    Returns
    -------
    list[np.ndarray]
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

    # project middle finger and ring finger on plane
    points = df_palm_landmarks.loc[[PALM_MIDDLE_FINGER, PALM_RING_FINGER]].values
    projections = project_point_to_plane(plane=plane, points=points)
    middle_finger_projection, ring_finger_projection = projections[0], projections[1]

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
    Assign visibility level to each finger landmark in a hand-tracking DataFrame.
    This function adds an additional column "visibility" to the DataFrame, indicating
    the visibility of each landmark based on its occlusion by other parts of the hand.

    Parameters
    ----------
    df_landmarks : pd.DataFrame
        DataFrame containing hand landmarks with columns: index, x, y, z.

    Returns
    -------
    None
        The function modifies the input DataFrame in place by adding a "visibility" column.
    """
    # Camera vector
    camera_vector = np.array([0.0, 0.0, -1.0]).reshape(-1, 1)

    # Create polygon using palm landmarks
    palm_plane = find_palm_plane(df_landmarks.loc[palm_landmarks])
    palm_polygon = construct_palm_polygon(df_landmarks.loc[palm_landmarks], palm_plane)

    # Coordinates of all points
    coords = ["x", "y", "z"]

    # Initialize visibility array
    visibility = np.ones(len(df_landmarks))  # 21

    # get dimentions of work
    points = df_landmarks.loc[:, coords].values
    point1 = points[[fc[0] for fc in finger_connections]]
    point2 = points[[fc[1] for fc in finger_connections]]
    n_points = points.shape[0]
    n_connections = len(point1)
    n_dim = points.shape[1]

    visibility = np.ones(n_points)

    # preparation
    point1 = point1.T.reshape(1, n_dim, n_connections)  # (1, 3, n_connections)
    point2 = point2.T.reshape(1, n_dim, n_connections)  # (1, 3, n_connections)

    # construct segments
    lines = point2 - point1  # (1, 3, n_connections)
    lines_norms = np.linalg.norm(lines, axis=1, keepdims=True)
    lines_norms = np.where(lines_norms <= 1e-3, 1.0, lines_norms)
    lines_normalized = lines / lines_norms  # (1, 3, n_connections)

    # get vectors to points
    vectors_to_points = (
        points.reshape(n_points, n_dim, 1) - point1
    )  # (n_points, 3, n_connections)

    # get lengths of projection vectors
    scales = np.sum(
        vectors_to_points * lines_normalized, axis=1, keepdims=True
    )  # (n_points, 1, n_connections)

    # find projection points
    projections = scales * lines_normalized + point1  # (n_points, 3, n_connections)

    # check if point is inside segment
    # -------------------------------------------------------
    # find segment
    segments = lines  # (1, 3, n_connections)
    segment_start = projections - point1  # (n_points, 3, n_connections)
    segment_end = projections - point2  # (n_points, 3, n_connections)

    # check relations between left side and right side vectors
    dot_check_start = np.sum(
        segment_start * segments, axis=1, keepdims=True
    )  # (n_points, 1, n_connections)
    dot_check_end = np.sum(
        segment_end * segments, axis=1, keepdims=True
    )  # (n_points, 1, n_connections)

    # check left side length and segment length
    segment_start_norm = np.linalg.norm(segment_start, axis=1, keepdims=True)
    segments_norm = np.linalg.norm(segments, axis=1, keepdims=True)

    # if left side is positive and right side is negative and projection length is smaller that segment lenght we are inside
    mask_between = (
        (dot_check_start >= 0)
        & (dot_check_end <= 0)
        & (segment_start_norm <= segments_norm)
    )  # (n_points, 1, n_connections)
    mask_between = mask_between.squeeze()  # (n_points, n_connections)
    # -------------------------------------------------------

    # find projection vectors
    projection_vectors = projections - points.reshape(n_points, n_dim, 1)
    projection_vectors_norm = np.linalg.norm(projection_vectors, axis=1, keepdims=True)
    projection_vectors_norm = np.where(
        projection_vectors_norm <= 1e-3, 1.0, projection_vectors_norm
    )
    projection_vectors /= projection_vectors_norm

    # check relation with camera vector
    cosine_values = np.sum(
        projection_vectors * camera_vector.reshape(1, -1, 1), axis=1, keepdims=True
    )  # (n_points, 1, n_connections)
    cosine_values = cosine_values.squeeze()

    # gain visibilities
    for point in range(n_points):
        cosine_values_point = cosine_values[point][mask_between[point]]
        if len(cosine_values_point) > 0:
            value = np.min(1.0 - cosine_values_point)
            visibility[point] = min(visibility[point], value)

    # Check palm occlusion for points not in palm landmarks
    not_palm_mask = np.isin(df_landmarks.index, palm_landmarks, invert=True)  # 21,
    palm_projections = project_point_to_plane(
        plane=palm_plane, points=points[not_palm_mask]
    )  # 21x3

    # if point is inside palm
    inside_palm_mask = is_inside_palm(palm_polygon, palm_plane, palm_projections)  # 21,

    palm_projection_vectors = (
        palm_projections[inside_palm_mask] - points[not_palm_mask][inside_palm_mask]
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


def landmarks_fusion(
    world_coordinates: list[pd.DataFrame], softmax_const: float = 1.0
) -> pd.DataFrame:
    """
    Combine information from different cameras and get the resulting set of landmarks.

    Parameters
    ----------
    world_coordinates: list[pd.DataFrame]
        World coordinates with assignment of visibility for each landmark.
        Each DataFrame contains 21 row with columns = [x, y, z, visibility]
    softmax_const: float = 1.0
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
