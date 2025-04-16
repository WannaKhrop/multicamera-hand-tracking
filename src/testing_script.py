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
from utils.constants import SOFTMAX_PARAM


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
def calculate_std(merged_frames: list[CameraFrame]) -> np.ndarray:
    """
    Calculate standard deviation of the merged frames.

    Parameters
    ----------
    merged_frames: list[CameraFrame]
        List of merged camera frames.

    Returns
    -------
    np.ndarray
        Standard deviation for each landmark.
    """
    all_landmarks = list()
    for frame in merged_frames:
        for hand in frame.landmarks.values():
            all_landmarks.append(hand[["x", "y", "z"]].values)

    # get targets
    landmarks_arr = np.array(all_landmarks)
    mean_value = np.mean(landmarks_arr, axis=0)

    # get distances to mean values
    distances = np.linalg.norm(landmarks_arr - mean_value, axis=2)
    # square distances
    distances = np.square(distances)
    # calculate std
    std = np.sqrt(np.mean(distances, axis=0))

    # return only interesting values
    return np.round(std, 4)


# calculate MSE-value for a gesture
def calculate_rmse(
    merged_frames: list[CameraFrame], y_true: np.ndarray, hand_id: str = "Right"
) -> np.ndarray:
    """
    Calculate the Root Mean Squared Error (RMSE) as distance between the predicted values from frames and the true values.

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
            A numpy array containing the RMSE for each frame.
    """
    all_landmarks = list()
    for frame in merged_frames:
        hand_values = frame.landmarks[hand_id]
        all_landmarks.append(hand_values[["x", "y", "z"]].values)

    # create matrix
    landmarks_arr = np.array(all_landmarks)

    # find distances
    distances = np.linalg.norm(landmarks_arr - y_true, axis=2)
    # get squared values
    distances = np.square(distances)
    # calculate MSE
    mse = np.mean(distances, axis=0)
    # get root of MSE
    rmse = np.sqrt(mse)

    # return only interesting values
    return np.round(rmse, 4)  # vector of 21 rmse values, for each landmark


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


def visualize_hand_sequence(
    predicted_sequence: list[pd.DataFrame], true_landmarks: pd.DataFrame
):
    """
    Visualize animated hand landmarks over time.

    Parameters
    ----------
    predicted_sequence: list[pd.DataFrame]
        List of predicted landmarks per frame.
    true_landmarks: pd.DataFrame
        List of ground truth landmarks per frame.
    """

    def create_scatter_traces(data: pd.DataFrame, color: str) -> list[go.Scatter3d]:
        # Scatter points
        scatter_data = go.Scatter3d(
            x=data.x.values,
            y=data.y.values,
            z=data.z.values,
            mode="markers+text",
            text=[str(idx) for idx in data.index],
            marker=dict(size=3, color=color),
            textfont=dict(size=6, color="blue"),
        )

        # Line connections
        connections_x, connections_y, connections_z = [], [], []
        for start_idx, end_idx in solutions.hands.HAND_CONNECTIONS:
            connections_x += [data.loc[start_idx].x, data.loc[end_idx].x, None]
            connections_y += [data.loc[start_idx].y, data.loc[end_idx].y, None]
            connections_z += [data.loc[start_idx].z, data.loc[end_idx].z, None]

        scatter_connections = go.Scatter3d(
            x=connections_x,
            y=connections_y,
            z=connections_z,
            mode="lines",
            line=dict(color=color, width=2),
        )

        return [scatter_data, scatter_connections]

    # Define layout
    layout = go.Layout(
        width=800,
        height=600,
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None]},
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        scene=dict(
            xaxis=dict(range=(0.0, 1.0)),
            yaxis=dict(range=(-1.0, 1.0)),
            zaxis=dict(range=(0.0, 1.5)),
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=2.0, z=1.0),
            camera=dict(eye=dict(x=1.0, y=1.0, z=2.0)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # First frame traces
    first_pred = create_scatter_traces(predicted_sequence[0], "black")
    first_true = create_scatter_traces(true_landmarks, "red")

    fig = go.Figure(data=first_pred + first_true, layout=layout, frames=[])
    frames = list()

    # Add animation frames
    for pred in predicted_sequence:
        traces = create_scatter_traces(pred, "black") + create_scatter_traces(
            true_landmarks, "red"
        )
        frame = go.Frame(data=traces)
        frames.append(frame)

    fig.frames = frames
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


def apply_one_test(
    path_to_dir: Path, visualize: bool = False
) -> dict[str, dict[str, list[float]]]:
    """
    Perform data processing for just one test directory.

    Parameters
    ----------
    path_to_dir: Path
        Path to test directory.
    visualize: bool = False
        Flag to visualize the result.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Data in the following format:
            Thread -> Metric -> Values for each landmark.
    """
    # get all logs
    files = set(glob(str(path_to_dir.joinpath("logs", "*.jsonl"))))
    fusion_log_file = str(
        path_to_dir.joinpath("logs", "log_Fusion.jsonl")
    )  #  actually, we do not need it
    real_data_file = str(path_to_dir.joinpath("landmarks.csv"))
    try:
        files.remove(fusion_log_file)  # do not include fusion log
    except Exception:
        pass

    # read all logs as one list
    camera_logs: list[CameraFrame] = list()
    for file in files:
        with open(file, "r") as log_file:
            for line in log_file:
                camera_logs.append(CameraFrame.from_dict(json.loads(line)))

    # read ground truth data
    real_values = pd.read_csv(real_data_file, sep=" ")

    # get all cameras we have
    cameras = set([log.camera_id for log in camera_logs])

    # apply simulation
    distributed_frames = distribute_frames(camera_logs)
    simulation_results = simulate_merging(
        distributed_frames=distributed_frames, cameras=cameras
    )

    # calculate STD and RMSE for each camera and Fusion
    fusion_data = dict()
    data_values: dict[str, dict] = dict()

    # get STD and RMSE
    fusion_data = {
        "std": calculate_std(merged_frames=simulation_results).tolist(),
        "rmse": calculate_rmse(
            merged_frames=simulation_results, y_true=real_values.values, hand_id="Right"
        ).tolist(),
    }

    data_values["Fusion"] = fusion_data

    for camera_id in cameras:
        # to collect camera data
        camera_data = dict()

        # get frames
        camera_frames = retrieve_frames_for_camera(
            distributed_frames, camera_id=camera_id
        )

        # get STD
        camera_data["std"] = calculate_std(merged_frames=camera_frames).tolist()

        # show MSE
        camera_data["rmse"] = calculate_rmse(
            merged_frames=camera_frames, y_true=real_values.values, hand_id="Right"
        ).tolist()

        # save results
        data_values[camera_id] = camera_data

    # in case of visualization
    if visualize:
        # extract a static gesture from merged frames
        gesture_landmarks = get_static_gesture(
            frames=simulation_results, hand_id="Right"
        )
        dynamic_gesture = [frame.landmarks["Right"] for frame in simulation_results]
        # plot this gesture
        visualize_hand(gesture_landmarks, real_values)
        visualize_hand_sequence(dynamic_gesture, real_values)

    return data_values


if True:
    # define basic path
    basic_path = Path("D:\\Project\\Testing\\All Runs")

    # define all log folders
    all_gestures = [
        "Palm Gesture",
        "Pointing Gesture",
        "Three Gesture",
        "Fist",
        "Fork",
        "Ok Gesture",
        "Thumb",
    ]
    all_views = ["Both Cameras", "Robot Camera", "Table Camera"]
    all_tests = ["Test#1", "Test#2", "Test#3", "Test#4", "Test#5"]
    landmarks_keys = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    ]

    df = pd.DataFrame(
        columns=["FusionParameter", "Gesture", "View", "TestID", "Camera", "Metric"]
        + landmarks_keys
    )
    idx = 0

    for gesture_dir in all_gestures:
        for view_dir in all_views:
            for test_dir in all_tests:
                # create a path
                path = basic_path.joinpath(gesture_dir, view_dir, test_dir)
                if not path.is_dir():
                    # directory does not exist
                    continue

                print(path)

                # proces just one test result
                test_results = apply_one_test(path_to_dir=path, visualize=False)

                # unwrap all the results and save them
                for cam_id in test_results.keys():
                    for metric in test_results[cam_id].keys():
                        # save data
                        df.loc[idx] = [
                            SOFTMAX_PARAM,
                            gesture_dir,
                            view_dir,
                            test_dir,
                            cam_id,
                            metric,
                        ] + test_results[cam_id][metric]
                        # update index
                        idx += 1

    df.to_csv("processed data.csv", index=False)

# in case of just one simple check
if False:
    path_to_data = Path("D:\\Project\\Testing\\All Runs\\Fist\\Both Cameras\\Test#2")
    results = apply_one_test(path_to_dir=path_to_data, visualize=True)
    # print results
    for key in results:
        print(50 * "=")
        print(key)
        for metric in results[key]:
            print(metric + ":", results[key][metric])
        print(50 * "=")
