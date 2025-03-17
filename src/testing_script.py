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
from pathlib import Path
import plotly.graph_objs as go
from mediapipe import solutions

# specific imports
from camera_thread.camera_frame import CameraFrame
from utils.constants import TIME_DELTA
from utils.fusion import DataMerger
from utils.mediapipe_world_model import MedapipeWorldTransformer
from hand_recognition.hand_recognizer import retrieve_from_depths
from utils.geometry import assign_visibility
from utils.coordinate_transformer import CoordinateTransformer

# get all logs
PATH_TO_DATA = Path(
    "C:\\Users\\khrop\\Desktop\\Test Results\\Fork\\Table Camera\\Test#5"
)  # puth path to data here
files = set(glob(str(PATH_TO_DATA.joinpath("logs", "*.jsonl"))))
fusion_log_file = str(
    PATH_TO_DATA.joinpath("logs", "log_Fusion.jsonl")
)  #  actually, we do not need it
real_data_file = str(PATH_TO_DATA.joinpath("landmarks.csv"))
try:
    files.remove(fusion_log_file)  # do not include fusion log
except Exception:
    pass


# get mean, std and max values from pd.DataFrame.describe
def retrieve_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get mean, std and max values from a DataFrame after calling describe()."""
    try:
        return df.loc[["mean", "std", "max"]]
    except Exception:
        return pd.DataFrame()


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


# simulate merging
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


#  retrieve results for a camera
def retrieve_frames_for_camera(
    distributed_frames: list[list[CameraFrame]],
    camera_id: str,
) -> list[CameraFrame]:
    """
    Retrieve frames for only one camera.

    Parameters
    ----------
    distributed_frames: list[list[CameraFrame]]
        List of lists of camera frames, each list contains frames close to each other.
    camera_id: str
        Camera ID to extract.

    Returns
    -------
    list[CameraFrame]
        List of camera frames.
    """
    # create ML Detectors to retrieve depths
    ml_detector = MedapipeWorldTransformer(camera_id=camera_id)

    # create coordinate transformer
    transformer = CoordinateTransformer()

    # process frames and select frames with camera_id only
    camera_frames: list[CameraFrame] = list()
    for frames in distributed_frames:
        for frame in frames:
            # get data from frame
            if frame is not None and frame.camera_id == camera_id:
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
            hand_depths = ml_detector(ml_detector, features=features)

            # convert to camera and then to world
            axes = ["x", "y", "z"]
            for hand, depths in zip(detected_hands, hand_depths):
                # camera coords
                retrieve_from_depths(
                    landmarks=detected_hands[hand],
                    depths=depths,
                    intrinsics=intrinsics,
                )

                # world coords
                detected_hands[hand].loc[:, axes] = transformer.camera_to_world(
                    camera_id=source,
                    points=detected_hands[hand].loc[:, axes].values,
                )

            # create a frame from it
            camera_frame = CameraFrame(
                timestamp=timestamp,
                camera_id=source,
                landmarks=detected_hands,
                intrinsics=None,
            )

            # store this frame
            camera_frames.append(camera_frame)

    return camera_frames


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

    # get targets
    landmarks_arr = np.array(all_landmarks)
    df = pd.DataFrame(np.std(landmarks_arr, axis=0), columns=["x", "y", "z"]).describe()

    # return only interesting values
    return retrieve_stats(df)


# calculate MAE-value for a gesture
def calculate_mae(
    merged_frames: list[CameraFrame], y_true: np.ndarray, hand_id: str = "Right"
) -> pd.DataFrame:
    """
    Calculate the Mean Absolute Error (MAE) between the predicted values from merged camera frames and the true values.

    Parameters
    ----------
        merged_frames: list[CameraFrame]
            A list of CameraFrame objects containing the predicted values from multiple cameras.
        y_true: np.ndarray
            A numpy array containing the true values for comparison.
        hand_id: str = "Right"
            Which hand must be evaluated.

    Returns
    -------
        np.ndarray
            A numpy array containing the MAE for each frame.
    """
    all_landmarks = list()
    for frame in merged_frames:
        hand_values = frame.landmarks[hand_id]
        all_landmarks.append(hand_values[["x", "y", "z"]].values)

    # create matrix
    landmarks_arr = np.array(all_landmarks)

    # find difference
    mae = np.mean(np.abs(landmarks_arr - y_true), axis=0)
    df = pd.DataFrame(mae, columns=["x", "y", "z"]).describe()

    # return only interesting values
    return retrieve_stats(df)


# calculate MSE-value for a gesture
def calculate_mse(
    merged_frames: list[CameraFrame], y_true: np.ndarray, hand_id: str = "Right"
) -> pd.DataFrame:
    """
    Calculate the Mean Squared Error (MSE) as distance between the predicted values from frames and the true values.

    Parameters
    ----------
        merged_frames: list[CameraFrame]
            A list of CameraFrame objects containing the predicted values from multiple cameras.
        y_true: np.ndarray
            A numpy array containing the true values for comparison.
        hand_id: str = "Right"
            Which hand must be evaluated.

    Returns
    -------
        np.ndarray
            A numpy array containing the MAE for each frame.
    """
    all_landmarks = list()
    for frame in merged_frames:
        hand_values = frame.landmarks[hand_id]
        all_landmarks.append(hand_values[["x", "y", "z"]].values)

    # create matrix
    landmarks_arr = np.array(all_landmarks)

    # find difference
    distances = np.linalg.norm(landmarks_arr - y_true, axis=2)
    mse = np.mean(distances, axis=0)
    df = pd.DataFrame(mse, columns=["distance"]).describe()

    # return only interesting values
    return retrieve_stats(df)


# visualize results
def visualize_hand(landmarks: pd.DataFrame, true_landmarks: pd.DataFrame):
    """
    Visualize hand landmarks.

    Parameters
    ----------
    landmarks: pd.DataFrame
        DataFrame containing landmarks.
    true_landmarks: pd.DataFrame
        Ground Truth Landmarks Positions.
    """

    # function for better visualization
    def create_scatter_plot(
        data: pd.DataFrame, color: str
    ) -> tuple[go.Scatter3d, go.Scatter3d]:
        # add scatter plot
        scatter_data = go.Scatter3d(
            x=data.loc[:].x.values,
            y=data.loc[:].y.values,
            z=data.loc[:].z.values,
            mode="markers+text",
            text=[str(idx) for idx in data.index],
            marker=dict(size=3, color=color),
            textfont=dict(size=6, color="blue"),
        )

        # add lines
        connections_x, connections_y, connections_z = [], [], []
        for start_idx, end_idx in solutions.hands.HAND_CONNECTIONS:
            connections_x += [
                data.loc[start_idx].x,
                data.loc[end_idx].x,
                None,
            ]
            connections_y += [
                data.loc[start_idx].y,
                data.loc[end_idx].y,
                None,
            ]
            connections_z += [
                data.loc[start_idx].z,
                data.loc[end_idx].z,
                None,
            ]

        scatter_connections = go.Scatter3d(
            x=connections_x,
            y=connections_y,
            z=connections_z,
            mode="lines",
            line=dict(color=color, width=2),
        )

        return scatter_data, scatter_connections

    # define layout for plotly
    custom_layout = go.Layout(
        autosize=False,
        width=800,
        height=600,
        showlegend=False,
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
            xaxis=dict(range=(0.0, 1.0), autorange=False),  # Set the x-axis limit
            yaxis=dict(range=(-1.0, 1.0), autorange=False),  # Set the y-axis limit
            zaxis=dict(range=(0.0, 1.5), autorange=False),  # Set the z-axis limit
            camera=dict(eye=dict(x=1.0, y=1.0, z=2.0)),
            aspectmode="manual",  # Fixes the aspect ratio
            aspectratio=dict(
                x=2.0, y=2.0, z=1.0
            ),  # Ensures aspect ratio remains constant
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # Tight margins for better visualization
    )

    # create figure
    fig = go.Figure(layout=custom_layout)

    # plot predicted data
    scatter_data_pred, scatter_conn_pred = create_scatter_plot(
        data=landmarks, color="black"
    )
    fig.add_trace(scatter_data_pred)
    fig.add_trace(scatter_conn_pred)

    # plot real gesture
    scatter_data_real, scatter_conn_real = create_scatter_plot(
        data=true_landmarks, color="red"
    )
    fig.add_trace(scatter_data_real)
    fig.add_trace(scatter_conn_real)

    # fig.write_image("gesture.svg")
    fig.show()


# get landmarks as average for a static gesture
def get_static_gesture(
    frames: list[CameraFrame], hand_id: str = "Right"
) -> pd.DataFrame:
    """
    Get average landmarks for a static gesture.

    Parameters
    ----------
        merged_frames: list[CameraFrame]
            A list of CameraFrame objects containing the predicted values from multiple cameras.
        hand_id: str = "Right"
            Which hand must be evaluated.

    Returns
    -------
        pd.DataFrame
            A DataFrame containing average landmarks.
    """
    all_landmarks = list()
    for frame in frames:
        hand_values = frame.landmarks[hand_id]
        all_landmarks.append(hand_values[["x", "y", "z"]].values)

    # create matrix and get average
    landmarks_arr = np.array(all_landmarks)
    landmarks_arr = np.mean(landmarks_arr, axis=0)

    # calculate landmarks
    landmarks = pd.DataFrame(landmarks_arr, columns=["x", "y", "z"])

    return landmarks


# read all logs as one list
camera_logs: list[CameraFrame] = list()
for file in files:
    with open(file, "r") as log_file:
        for line in log_file:
            camera_logs.append(CameraFrame.from_dict(json.loads(line)))

# read fusion log
# with open(fusion_log_file, "r") as log_file:
#    fusion_logs = [CameraFrame.from_dict(json.loads(line)) for line in log_file]

# read ground truth data
real_values = pd.read_csv(real_data_file, sep=" ")

# get all cameras we have
cameras = set([log.camera_id for log in camera_logs])

# apply simulation
distributed_frames = distribute_frames(camera_logs)
simulation_results = simulate_merging(
    distributed_frames=distributed_frames, cameras=cameras
)

# show STD
print("Standard Deviation")
print(calculate_std(merged_frames=simulation_results))
print()

# show MAE
print("Mean Absolute Error")
print(
    calculate_mae(
        merged_frames=simulation_results, y_true=real_values.values, hand_id="Right"
    )
)
print()

# show MSE
print("Mean Squared Error")
print(
    calculate_mse(
        merged_frames=simulation_results, y_true=real_values.values, hand_id="Right"
    )
)
print()

for camera_id in cameras:
    print(60 * "=")
    print()

    # get frames
    camera_frames = retrieve_frames_for_camera(distributed_frames, camera_id=camera_id)

    print("Camera", camera_id)

    print()
    # show STD
    print("Standard Deviation")
    print(calculate_std(merged_frames=camera_frames))
    print()

    # show MAE
    print("Mean Absolute Error")
    print(
        calculate_mae(
            merged_frames=camera_frames, y_true=real_values.values, hand_id="Right"
        )
    )
    print()

    # show MSE
    print("Mean Squared Error")
    print(
        calculate_mse(
            merged_frames=camera_frames, y_true=real_values.values, hand_id="Right"
        )
    )
    print()

    # close reporting
    print(60 * "=")

# extract a static gesture from merged frames
gesture_landmarks = get_static_gesture(frames=simulation_results, hand_id="Right")

# plot this gesture
visualize_hand(gesture_landmarks, real_values)
