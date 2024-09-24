"""
This module can be extended in the future to get preciser hand coordinates using neural networks.

Author: Ivan Khrop
Date: 24.09.2024
"""

import mediapipe as mp
import numpy as np
import warnings
from camera_thread.rs_thread import CameraThreadRS
from utils.constants import PATH_TO_MODEL
from time import sleep

# functions to process images
from camera_thread.camera import camera
from hand_recognition.hand_recognizer import to_numpy_ndarray
from utils.mediapipe_world_model import MedapipeWorldTransformer

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

    my_cam = camera(available_cameras[0][0], available_cameras[0][1])
    landmarker = HandLandmarker.create_from_options(options)
    transformer = MedapipeWorldTransformer()

    input("Start capturing: ")

    # define events and data
    for _ in range(2000):
        # get frame
        color_frame = my_cam.take_picture_and_return_color()
        depth_frame = my_cam.get_last_depth_frame()
        height, width, _ = color_frame.shape

        # process frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        mp_results = landmarker.detect(mp_image)

        # check model
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

            # get features
            features = np.hstack([landmarks, column])
            features = np.hstack([features, np.array([height, width])]).reshape(1, -1)
            world_landmarks = to_numpy_ndarray(world_landmarks)

            # apply model
            predictions = transformer.predict(features=features, shape=landmarks.shape)

            # compare
            print("Difference: ", np.linalg.norm(predictions - world_landmarks))
            print("Average difference: ", np.mean(predictions - world_landmarks))
            print("Max difference: ", np.max(abs(predictions - world_landmarks)))

        print(40 * "=")
        sleep(1.0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
