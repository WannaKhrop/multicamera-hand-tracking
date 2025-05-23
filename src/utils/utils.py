"""
Module contains helpful function that are used in this project.

Author: Ivan Khrop
Date: 21.07.2024
"""
import numpy as np
import pandas as pd
import cv2
from typing import Iterable, Sequence
from time import time
import json
from threading import Lock
from functools import wraps
from typing import TypeVar, ParamSpec, Callable, Generic
from camera_thread.camera_frame import CameraFrame

from utils.constants import (
    PATH_TO_VIDEOS,
    CAMERA_RESOLUTION_WIDTH,
    CAMERA_RESOLUTION_HEIGHT,
    PATH_TO_LOGS,
)

# for decorators
F_Spec = ParamSpec("F_Spec")
F_Return = TypeVar("F_Return")


def merge_sorted_lists(
    list1: Iterable[tuple[int, str, dict[str, pd.DataFrame]]],
    list2: Iterable[tuple[int, str, dict[str, pd.DataFrame]]],
) -> Iterable[tuple[int, str, dict[str, pd.DataFrame]]]:
    """
    Merge two sorted lists according to timestamps in one sequence

    Parameters
    ----------
    list1: Iterable[tuple[int, str, np.array, np.array, rs.pyrealsense2.intrinsics]]
        First list to be merged. Each element is (timestamp, camera_id, frame, depth_frame, intrinsics)
    list2: Iterable[tuple[int, str, np.array, np.array, rs.pyrealsense2.intrinsics]]
        Second list to be merged. Each element is (timestamp, camera_id, frame, depth_frame, intrinsics)

    Returns
    -------
    merged_list: Iterable[tuple[int, np.array, np.array, rs.pyrealsense2.intrinsics]]
        Resulting list with elements sorted in ascending order accoring to timestamps.
        Each element is (timestamp, camera_id, frame, depth_frame, intrinsics)
    """

    i: int = 0  # type: ignore
    j: int = 0  # type: ignore
    merged_list = []

    list1 = list(list1)
    list2 = list(list2)

    # Loop until one of the lists is exhausted
    while i < len(list1) and j < len(list2):
        if list1[i][0] < list2[j][0]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1

    # Append remaining elements from list1 if any
    while i < len(list1):
        merged_list.append(list1[i])
        i += 1

    # Append remaining elements from list2 if any
    while j < len(list2):
        merged_list.append(list2[j])
        j += 1

    return merged_list


def umeyama(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Umeyama algorithm returns linear transformation matrix [R|t].
    Where:
        R - rotation matrix
        t - translation vector

    Mathematical justification of this algorithm can be found here[https://web.stanford.edu/class/cs273/refs/umeyama.pdf]

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


def softmax(data: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply [Softmax](https://en.wikipedia.org/wiki/Softmax_function) with temperature for each row of data.

    Parameters
    ----------
    data: np.ndarray
        Matrix where softmax will be applied for each row.
    temperature: float = 1.0
        Temperature that is used for scaling.

    Returns
    -------
    np.ndarray
        Result after softmax.
    """
    # some preprocssing for stabilization
    max_vals = np.max(data, axis=1, keepdims=True)
    shifted_matrix = temperature * (data - max_vals)

    # Exponentiate the shifted values
    exp_vals = np.exp(shifted_matrix)

    # Sum of exponentiated values for each row
    sum_exp_vals = np.sum(exp_vals, axis=1, keepdims=True)

    # Divide exponentiated values by the sum of exponentiated values for each row
    softmax_matrix = exp_vals / sum_exp_vals

    return softmax_matrix


def make_video(name: str, frames: Iterable[np.ndarray]):
    """
    Create a video from a sequence of frames.

    Parametrs
    ---------
    name : str
        The name of the output video file (without extension).
    frames : Iterable[np.ndarray]
        An iterable of frames (numpy arrays) to be included in the video.
    """
    # check if there is anything
    if len(list(frames)) == 0:
        return

    # configure video parameters
    video_name: str = str(PATH_TO_VIDEOS.joinpath(name + ".avi"))
    codec = cv2.VideoWriter.fourcc(*"XVID")
    size: Sequence[int] = (CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT)
    video = cv2.VideoWriter(filename=video_name, fourcc=codec, fps=7.0, frameSize=size)

    # save video
    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def write_logs(frames: list[CameraFrame], camera_id: str):
    """
    Write logs to the file.

    Parameters
    ----------
    frames: list[CameraFrame]
        List of frames that will be written to the file.
    camera_id: str
        Camera id that will be used for the file name.
    """
    # create path to the file
    path_to_file = PATH_TO_LOGS.joinpath(f"log_{camera_id}.jsonl")
    # create dictionary with data
    data_to_save = [frame.to_dict() for frame in frames]
    # save it to the file
    with open(path_to_file, "w") as file:
        for item in data_to_save:
            file.write(json.dumps(item) + "\n")


class TimeChecker(Generic[F_Spec, F_Return]):
    """
    Class represents a decorator to check average runtime of a decorated funciton.

    Attributes
    ----------
    n_calls: int
        Number of calls of a function.
    total_time: float
        Total time spent for running function.
    call: Callable[F_Spec, F_Return]
        Function that will be called.
    """

    n_calls: int
    total_time: float
    call: Callable[F_Spec, F_Return]

    def __init__(self, call: Callable[F_Spec, F_Return]):
        self.n_calls = 0
        self.total_time = 0.0
        self.call = call

    def __call__(self, *args: F_Spec.args, **kwargs: F_Spec.kwargs) -> F_Return:
        # add call counter
        self.n_calls += 1
        # start time, call function, save run time
        start = time()
        call_result = self.call(*args, **kwargs)
        self.total_time += time() - start
        # return result
        return call_result

    def __del__(self):
        avg_time = round(self.total_time / self.n_calls, 3) if self.n_calls > 0 else 0.0
        print(40 * "=")
        print("Report for function:", self.call.__name__)
        print("Total amount of calls:", self.n_calls)
        print(f"Average time: {avg_time} sec.")
        print(40 * "=")


def thread_safe(call: Callable[F_Spec, F_Return]) -> Callable[F_Spec, F_Return]:
    """
    Decorator to create thread safe function calls for shared resources.

    Parameters
    ----------
    call: Callable[F_Spec, F_Return]
        Function that will be called.

    Returns
    -------
    Callable[F_Spec, F_Return]
        Wrapper for the call-function.
    """
    # create locker for thread safety
    lock = Lock()

    @wraps(call)
    def wrapper(*args: F_Spec.args, **kwargs: F_Spec.kwargs) -> F_Return:
        # wait utill locker is free to call the function
        with lock:
            return call(*args, **kwargs)

    return wrapper
