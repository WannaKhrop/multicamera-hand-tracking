"""
This module can be extended in the future to get preciser hand coordinates using neural networks.

Author: Ivan Khrop
Date: 24.09.2024
"""

import numpy as np
import warnings
from camera_thread.rs_ml_thread import MLCameraThreadRS
from utils.constants import PATH_TO_DATA_FOLDER
from threading import Event
from utils.constants import PROB_PARAM_DISANCE, PROB_PARAM_ZERO


def main():
    # read available cameras
    available_cameras = MLCameraThreadRS.returnCameraIndexes()
    # check cameras
    assert (
        len(available_cameras) > 0
    ), "No cameras are available. Test can not be passed."

    # define events and data
    close_threads = Event()
    threads: dict[str, MLCameraThreadRS] = dict()

    for camera_name, camera_id in available_cameras:
        threads[camera_id] = MLCameraThreadRS(camera_name, camera_id, close_threads)

    # !!! TODO !!!
    # add event to start threads
    input("Start capturing: ")

    for camera_id in threads:
        threads[camera_id].start()

    # wait till the end
    input("Input to interrupt all thread: ")
    close_threads.set()

    # collect data
    data_in, data_out = list(), list()
    for camera_id in threads:
        features, targets = list(), list()
        for elem_rel_depths, elem_depths in threads[camera_id].target:
            # add clear data
            targets.append(elem_depths)

            # corrupt depths a bit
            corrupted_depths = elem_depths.copy()
            corrupted_depths += np.random.normal(
                loc=0.0, scale=0.005, size=corrupted_depths.shape
            )

            bernoulli_vector_dist = np.random.binomial(
                n=1, p=PROB_PARAM_DISANCE, size=corrupted_depths.shape
            )
            corrupted_depths[bernoulli_vector_dist] += np.abs(
                np.random.normal(loc=1.5, scale=2.0, size=corrupted_depths.shape)
            )

            bernoulli_vector_zero = np.random.binomial(
                n=1, p=PROB_PARAM_ZERO, size=corrupted_depths.shape
            )
            corrupted_depths[bernoulli_vector_zero] = 0.0

            # save corrupted
            features.append(np.hstack([elem_rel_depths, corrupted_depths]))

        data_in.append(np.array(features))
        data_out.append(np.array(targets))

    # make matrixes
    data_x = np.vstack(data_in)
    data_y = np.vstack(data_out)

    print(data_x.shape)
    print(data_y.shape)

    # save data
    np.save(str(PATH_TO_DATA_FOLDER.joinpath("features.npy")), data_x)
    np.save(str(PATH_TO_DATA_FOLDER.joinpath("targets.npy")), data_y)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
