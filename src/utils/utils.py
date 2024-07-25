"""
Module contains helpful function that are used in this project.

Author: Ivan Khrop
Date: 21.07.2024
"""
import numpy as np


def merge_sorted_lists(
    list1: list[tuple[int, np.array]],
    list2: list[tuple[int, np.array]],
    source1: int,
    source2: int,
) -> list[tuple[int, int, np.array]]:
    """
    Merge two sorted lists according to timestamps in one sequence

    Parameters
    ----------
    list1: list[tuple[int, np.array]]
        First list to be merged. Each element is (timestamp, frame)
    list2: list[tuple[int, np.array]]
        Second list to be merged. Each element is (timestamp, frame)
    source1: int
        Camera number which captured list1
    source1: int
        Camera number which captured list2

    Returns
    -------
    merged_list: list[tuple[int, int, np.array]]
        Resulting list with elements sorted in ascending order accoring to timestamps.
        Each element is (timestamp, source, frame )
    """

    i: int = 0  # type: ignore
    j: int = 0  # type: ignore
    merged_list = []

    # Loop until one of the lists is exhausted
    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            merged_list.append((list1[i][0], source1, list[i][1]))
            i += 1
        else:
            merged_list.append((list2[j][0], source2, list[j][1]))
            j += 1

    # Append remaining elements from list1 if any
    while i < len(list1):
        merged_list.append((list1[i][0], source1, list[i][1]))
        i += 1

    # Append remaining elements from list2 if any
    while j < len(list2):
        merged_list.append((list2[j][0], source2, list[j][1]))
        j += 1

    return merged_list


def umeyama(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Umeyama algorithm returns linear transformation matrix [R|t].
    Where:
        R - rotation matrix
        t - translation vector

    Mathimatical justification of this algorithm can be found here[https://web.stanford.edu/class/cs273/refs/umeyama.pdf]

    Parameters
    ----------
    X: np.ndarray [N, d]
        Cloud of points in one coordinate system. N - number of points, d - dimensionality
    Y: np.ndarray [N, d]
        Cloud of points in target coordinate system. N - number of points, d - dimensionality

    Returns
    -------
    M: np.ndarray [d + 1, d + 1]
        Transfomation matrix [R|t].
    """

    n, m = X.shape

    # Compute centroids
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)

    # Center the points
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute covariance matrix
    covariance_matrix = np.dot(Y_centered.T, X_centered) / n

    # Perform SVD
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Compute rotation matrix
    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0
    if d:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]

    R = np.dot(U, Vt)

    # Compute scaling factor
    scale = 1.0

    # Compute translation vector
    t = centroid_Y.T - scale * np.dot(R, centroid_X.T)

    # Construct the transformation matrix
    transformation_matrix = np.identity(m + 1)
    transformation_matrix[:m, :m] = scale * R
    transformation_matrix[:m, m] = t

    return transformation_matrix


def linear_transfomation(X_data: np.ndarray, Y_data: np.ndarray) -> np.ndarray:
    """
    Simple minimization algorithm returns linear transformation matrix [R|t] between two points clouds.
    Where:
        R - rotation matrix
        t - translation vector

    Parameters
    ----------
    X: np.ndarray [d, N]
        Cloud of points in one coordinate system. N - number of points, d - dimensionality
    Y: np.ndarray [d, N]
        Cloud of points in target coordinate system. N - number of points, d - dimensionality

    Returns
    -------
    M: np.ndarray [d + 1, d + 1]
        Transfomation matrix [R|t].
    """
    d, n = X_data.shape

    matrix = np.eye(d + 1)

    X = np.vstack([X_data, np.ones((1, n))])

    # results
    inversed = np.linalg.inv(np.dot(X, X.T))
    M = Y_data.dot(X.T).dot(inversed)

    matrix[:d] = M

    return matrix
