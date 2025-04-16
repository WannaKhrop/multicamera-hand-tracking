import os
import sys
import numpy as np

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from camera_thread.camera import camera  # noqa: E402
from camera_thread.rs_thread import CameraThreadRS  # noqa: E402
from hand_recognition.hand_recognizer import (  # noqa: E402
    CAMERA_RESOLUTION_HEIGHT,
    CAMERA_RESOLUTION_WIDTH,
)


def test_async_depth_detection(make_test: bool = False):
    # there is no need to run test (skip for commit without cameras)
    if not make_test:
        assert True, "Test is skipped."
    else:
        # find available cameras
        arr = CameraThreadRS.returnCameraIndexes()
        print(arr)

        assert len(arr) > 0, "No cameras are available. Test can not be passed."

        # define camera
        my_cam = camera(device_name=arr[0][0], device_id=arr[0][1])

        # number of checks
        N = 20

        # generate list of pairs x_pixel, y_pixel
        checks = [
            (
                np.random.randint(CAMERA_RESOLUTION_WIDTH),
                np.random.randint(CAMERA_RESOLUTION_HEIGHT),
            )
            for _ in range(N)
        ]

        # define some lists to store results
        async_data, sync_results, async_results = list(), list(), list()

        # take pictures and get results synchronously
        for x_pixel, y_pixel in checks:
            # get data
            frame = my_cam.take_picture_and_return_color()
            depth_data = my_cam.get_last_depth_frame()
            intrinsics = my_cam.get_last_intrinsics()

            # save async
            async_data.append((frame, depth_data, intrinsics))

            # get sync results
            sync_result = my_cam.get_depth_data_from_pixel(x_pixel, y_pixel)
            sync_results.append(sync_result)

        # now go over historical data and calculate
        for i, elem in enumerate(checks):
            # get data
            x_pixel, y_pixel = elem
            _, depth_frame, intrinsic = async_data[i]
            # detect camera coordinates asynchronously
            depth = camera.get_depth(x_pixel, y_pixel, depth_frame=depth_frame)
            async_result = camera.get_coordinates_for_depth(
                x_pixel, y_pixel, depth, intrinsic
            )

            async_results.append(async_result)

        # compare results
        sync_result = np.vstack(sync_results)
        async_result = np.vstack(async_results)

        assert np.linalg.norm(async_result - sync_result) < 1e-3
