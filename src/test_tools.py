"""
This module contains functions for testing the application using data from logs.

Author: Ivan Khrop
Date: 14.01.2025
"""
# standard imports
import json
from glob import glob
import numpy as np
import pandas as pd

# specific imports
from camera_thread.camera_frame import CameraFrame
from utils.constants import TIME_DELTA, PATH_TO_LOGS
from utils.fusion import DataMerger
from utils.mediapipe_world_model import MedapipeWorldTransformer
from hand_recognition.hand_recognizer import retrieve_from_depths
from utils.geometry import assign_visibility
from utils.coordinate_transformer import CoordinateTransformer

# get all logs
files = set(glob(str(PATH_TO_LOGS.joinpath("*.jsonl"))))
fusion_log_file = str(PATH_TO_LOGS.joinpath("log_Fusion.jsonl"))
try:
    files.remove(fusion_log_file)  # do not include fusion log
except Exception:
    pass


# function to organize frames for merging
def distribute_frames(frames: list[CameraFrame]) -> list[list[CameraFrame]]:
    """
    Redistribute available camera frames so that close frame are in one list with respect to TIME_DELTA.

    Parameters
    ----------
    frames: list[CameraFrame]
        List of camera frames to be distributed.

    Returns
    -------
    list[list[CameraFrame]]
        List of lists of camera frames, each list contains frames close to each other.
    """
    # sort frames by timestamp
    sorted_frames = sorted(frames, key=lambda frame: frame.timestamp)

    # distribute frames
    distributed_frames = list()
    # add initial frame
    current_frames = [sorted_frames[0]]
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i].timestamp - current_frames[0].timestamp <= TIME_DELTA:
            current_frames.append(sorted_frames[i])
        else:
            distributed_frames.append(current_frames)
            current_frames = [sorted_frames[i]]

    return distributed_frames


# read all logs as one list
camera_logs: list[CameraFrame] = list()
for file in files:
    with open(file, "r") as log_file:
        for line in log_file:
            camera_logs.append(CameraFrame.from_dict(json.loads(line)))

# read fusion log
with open(fusion_log_file, "r") as log_file:
    fusion_logs = [CameraFrame.from_dict(json.loads(line)) for line in log_file]

# get all cameras we have
cameras = set([log.camera_id for log in camera_logs])


def simulate_merging(
    distributed_frames: list[list[CameraFrame]],
    cameras: set[str],
) -> list[CameraFrame]:
    """
    Simulate merging of camera frames.

    Parameters
    ----------
    distributed_frames: list[list[CameraFrame]]
        List of lists of camera frames, each list contains frames close to each other.
    cameras: set[str]
        Set of camera used to capture frames.

    Returns
    -------
    list[CameraFrame]
        List of merged camera frames.
    """
    # create DataMerger
    data_merger = DataMerger(time_delta=TIME_DELTA)

    # create ML Detectors to retrieve depths
    ml_detectors = dict()
    for camera_id in cameras:
        ml_detectors[camera_id] = MedapipeWorldTransformer(camera_id=camera_id)

    # create coordinate transformer
    transformer = CoordinateTransformer()

    # process frames
    for frames in distributed_frames:
        for frame in frames:
            # get data from frame
            if frame is not None:
                timestamp, source, detected_hands, intrinsics = frame.as_tuple()
            else:
                continue

            # process each hand
            features = np.empty(shape=(0, 42))
            for hand in detected_hands:
                # extract features
                features_hand = np.hstack(
                    [
                        detected_hands[hand].z.values.copy(),
                        detected_hands[hand].depth.values.copy(),
                    ]
                )
                features = np.vstack([features, features_hand])
                # drop depth column as we do not need it anymore
                detected_hands[hand].drop(columns=["depth"], inplace=True)

            # predict real depths using ml
            hand_depths = ml_detectors[source](ml_detectors[source], features=features)

            # convert to camera and then to world
            axes = ["x", "y", "z"]
            for hand, depths in zip(detected_hands, hand_depths):
                # camera coords
                retrieve_from_depths(
                    landmarks=detected_hands[hand],
                    depths=depths,
                    intrinsics=intrinsics,
                )

                # assign visibility
                assign_visibility(detected_hands[hand])

                # world coords
                detected_hands[hand].loc[:, axes] = transformer.camera_to_world(
                    camera_id=source,
                    points=detected_hands[hand].loc[:, axes].values,
                )

            # make fusion
            camera_frame = CameraFrame(
                timestamp=timestamp,
                camera_id=source,
                landmarks=detected_hands,
                intrinsics=None,
            )

            data_merger.add_time_frame(camera_frame)

        # do fusion
        if len(frames) > 0:
            data_merger.make_fusion(data_merger)

    return data_merger.fusion_results


# apply simulation
distributed_frames = distribute_frames(camera_logs)
simulation_results = simulate_merging(
    distributed_frames=distributed_frames, cameras=cameras
)


# calculate STD-value for a gesture
def calculate_std(merged_frames: list[CameraFrame]) -> pd.DataFrame:
    """
    Calculate standard deviation of the merged frames.

    Parameters
    ----------
    merged_frames: list[CameraFrame]
        List of merged camera frames.

    Returns
    -------
    np.ndarray
        Standard deviation of the merged frames.
    """
    all_landmarks = list()
    for frame in merged_frames:
        for hand in frame.landmarks.values():
            all_landmarks.append(hand[["x", "y", "z"]].values)

    landmarks_arr = np.array(all_landmarks)
    return pd.DataFrame(np.std(landmarks_arr, axis=0), columns=["x", "y", "y"])


# calculate MAE-value for a gesture
def calculate_mae(merged_frames: list[CameraFrame], y_true: np.ndarray) -> pd.DataFrame:
    """
    Calculate the Mean Absolute Error (MAE) between the predicted values from merged camera frames and the true values.

    Parameters
    ----------
        merged_frames: list[CameraFrame]
            A list of CameraFrame objects containing the predicted values from multiple cameras.
        y_true: np.ndarray
            A numpy array containing the true values for comparison.
    Returns
    -------
        np.ndarray
            A numpy array containing the MAE for each frame.
    """
    all_landmarks = list()
    for frame in merged_frames:
        for hand in frame.landmarks.values():
            all_landmarks.append(hand[["x", "y", "z"]].values)

    # create matrix
    landmarks_arr = np.array(all_landmarks)

    # find difference
    mae = np.mean(np.abs(landmarks_arr - y_true), axis=0)
    return pd.DataFrame(mae, columns=["x", "y", "y"])


# show STD
print(calculate_std(merged_frames=simulation_results))
