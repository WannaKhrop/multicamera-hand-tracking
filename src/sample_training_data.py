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
    print(available_cameras)
    # check cameras
    assert len(available_cameras) == 1, "Sampling is possible for one camera only !!!"

    # define events and data
    close_threads = Event()
    threads: dict[str, MLCameraThreadRS] = dict()

    for camera_name, camera_id in available_cameras:
        threads[camera_id] = MLCameraThreadRS(camera_name, camera_id, close_threads)

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
        for elem_features, elem_depths in threads[camera_id].target:
            # add clear data
            targets.append(elem_depths)

            # corrupt depths a bit !!! OPTIONAL ???
            corrupted_depths = elem_depths.copy()
            corrupted_depths += np.random.normal(
                loc=0.0, scale=0.01, size=corrupted_depths.shape
            )

            # add some extra distance
            bernoulli_vector_dist = np.random.binomial(
                n=1, p=PROB_PARAM_DISANCE, size=corrupted_depths.shape
            )
            bernoulli_vector_dist = np.array(bernoulli_vector_dist, dtype=bool)
            corrupted_depths[bernoulli_vector_dist] += np.abs(
                np.random.normal(loc=1.5, scale=2.0, size=sum(bernoulli_vector_dist))
            )

            # some values to zero
            bernoulli_vector_zero = np.random.binomial(
                n=1, p=PROB_PARAM_ZERO, size=corrupted_depths.shape
            )
            bernoulli_vector_zero = np.array(bernoulli_vector_zero, dtype=bool)
            corrupted_depths[bernoulli_vector_zero] = 0.0

            # save corrupted
            features.append(np.hstack([elem_features, corrupted_depths]))

        data_in.append(np.array(features))
        data_out.append(np.array(targets))

    # make matrixes
    data_x = np.vstack(data_in)
    data_y = np.vstack(data_out)

    # we collect a lot of data all the time, so we can just add it in the existing file !!!
    features_arr = np.empty((0, data_x.shape[1]))
    targets_arr = np.empty((0, data_y.shape[1]))
    # add new data and as vertical stack
    features_arr = np.vstack([features_arr, data_x])
    targets_arr = np.vstack([targets_arr, data_y])

    # print results
    print(features_arr.shape)
    print(targets_arr.shape)

    # save data
    np.save(
        str(PATH_TO_DATA_FOLDER.joinpath(f"{camera_id}_features.npy")), features_arr
    )
    np.save(str(PATH_TO_DATA_FOLDER.joinpath(f"{camera_id}_targets.npy")), targets_arr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
