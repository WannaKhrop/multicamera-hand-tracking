"""
This module can be extended in the future to get preciser hand coordinates using neural networks.

Author: Ivan Khrop
Date: 24.09.2024
"""

import mediapipe as mp
import numpy as np
import warnings
from camera_thread.rs_thread import CameraThreadRS
from utils.constants import PATH_TO_MODEL, PATH_TO_DATA_FOLDER

# functions to process images
from camera_thread.camera import camera
from hand_recognition.hand_recognizer import to_numpy_ndarray

# define types
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# usage of hands - model !!!
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=PATH_TO_MODEL),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE,
)


def main():
    # read available cameras
    available_cameras = CameraThreadRS.returnCameraIndexes()
    # check cameras
    assert (
        len(available_cameras) > 0
    ), "No cameras are available. Test can not be passed."

    # define parts
    my_cam = camera(available_cameras[0][0], available_cameras[0][1])
    landmarker = HandLandmarker.create_from_options(options)

    data_in = list()
    data_out = list()

    input("Start capturing: ")

    # define events and data
    for _ in range(3000):
        # get frame
        color_frame = my_cam.take_picture_and_return_color()
        depth_frame = my_cam.get_last_depth_frame()
        height, width, _ = color_frame.shape

        # process frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        mp_results = landmarker.detect(mp_image)

        # get features and targets
        for landmarks, world_landmarks in zip(
            mp_results.hand_landmarks, mp_results.hand_world_landmarks
        ):
            landmarks = to_numpy_ndarray(landmarks)
            depths: list[float] = list()

            # get depths
            for landmark in landmarks:
                x, y = landmark[0], landmark[1]
                x_pixel = int(width * x)
                y_pixel = int(y * height)

                depths.append(
                    camera.get_depth(
                        x_pixel=x_pixel, y_pixel=y_pixel, depth_frame=depth_frame
                    )
                )

            column = np.array(depths).reshape(-1, 1)

            # features and targets
            features = np.hstack([landmarks, column]).reshape(-1)
            features = np.hstack([features, np.array([height, width])])
            world_landmarks = to_numpy_ndarray(world_landmarks)

            # store them
            data_in.append(features)
            data_out.append(world_landmarks.reshape(-1))

        print(40 * "=")

    # make matrixes
    data_x = np.array(data_in)
    data_y = np.array(data_out)

    print(data_x.shape)
    print(data_y.shape)

    # save data
    np.save(str(PATH_TO_DATA_FOLDER.joinpath("features.npy")), data_x)
    np.save(str(PATH_TO_DATA_FOLDER.joinpath("targets.npy")), data_y)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
