"""
This module can be extended in the future to get preciser hand coordinates using neural networks.

Author: Ivan Khrop
Date: 24.09.2024
"""

import numpy as np
import warnings
from camera_thread.rs_ml_thread import MLCameraThreadRS
from time import sleep

# functions to process images
from camera_thread.camera import camera
from hand_recognition.hand_recognizer import to_numpy_ndarray
from utils.mediapipe_world_model import MedapipeWorldTransformer
from hand_recognition.HolisticLandmarker import HolisticLandmarker


def main():
    # read available cameras
    available_cameras = MLCameraThreadRS.returnCameraIndexes()
    # check cameras
    assert (
        len(available_cameras) > 0
    ), "No cameras are available. Test can not be passed."

    print("Available cameras: ", list(enumerate(available_cameras)))
    cam_id = int(input("Choose camera id: "))

    my_cam = camera(available_cameras[cam_id][0], available_cameras[cam_id][1])
    landmarker = HolisticLandmarker()
    transformer = MedapipeWorldTransformer(camera_id=available_cameras[cam_id][1])

    sleep(1.0)
    input("Start capturing: ")

    # define events and data
    for _ in range(2000):
        # get frame
        color_frame = my_cam.take_picture_and_return_color()
        depth_frame = my_cam.get_last_depth_frame()
        intrinsics = my_cam.get_last_intrinsics()

        # process frame
        mp_results = landmarker.process_image(landmarker, color_frame)

        # for each hand get depths and
        if mp_results.left_hand_landmarks is not None:
            landmarks = to_numpy_ndarray(mp_results.left_hand_landmarks)
            # get depth data
            rel_depths, depths = MLCameraThreadRS.process_hand(
                landmarks, depth_frame, intrinsics
            )

            if (depths > 1e-1).all():
                # get features
                features = np.hstack([rel_depths, depths]).reshape(1, -1)
                # apply model
                predictions = transformer.predict(
                    transformer, features=features, shape=depths.shape
                ).squeeze()

                # compare
                print("Difference: ", np.linalg.norm(predictions - depths))
                print("Average difference: ", np.mean(abs(predictions - depths)))
                print("Max difference: ", np.max(abs(predictions - depths)))
                print("Predict: ", predictions)
                print("Depths: ", depths)

        # for each hand get depths and
        if mp_results.right_hand_landmarks is not None:
            landmarks = to_numpy_ndarray(mp_results.right_hand_landmarks)
            # get depth data
            rel_depths, depths = MLCameraThreadRS.process_hand(
                landmarks, depth_frame, intrinsics
            )

            if (depths > 1e-1).all():
                # get features
                features = np.hstack([rel_depths, depths]).reshape(1, -1)
                # apply model
                predictions = transformer.predict(
                    transformer, features=features, shape=depths.shape
                ).squeeze()

                # compare
                print("Difference: ", np.linalg.norm(predictions - depths))
                print("Average difference: ", np.mean(abs(predictions - depths)))
                print("Max difference: ", np.max(abs(predictions - depths)))
                print("Predict: ", predictions)
                print("Depths: ", depths)

        print(40 * "=")
        sleep(1.0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
