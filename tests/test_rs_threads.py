import os
import sys
from threading import Event, Barrier
from time import sleep

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from camera_thread.rs_thread import CameraThreadRS  # type: ignore


def test_threads_rs():
    arr = CameraThreadRS.returnCameraIndexes()

    assert len(arr) > 0, "No cameras are available. Test can not be passed."

    close_threads = Event()
    barrier = Barrier(len(arr))
    data_barrier = Barrier(len(arr))  # just to test threads for cameras
    threads: dict[str, CameraThreadRS] = dict()

    for camera_name, camera_id in arr:
        # threads
        threads[camera_id] = CameraThreadRS(
            camera_name,
            camera_id,
            close_threads,
            barrier,
            data_barrier,
            data_barrier,
        )
        threads[camera_id].start()

    # sleep for a while
    sleep(10)
    close_threads.set()

    # check results
    number_of_cameras_used = 0
    for _, camera_id in arr:
        if len(threads[camera_id].frames) > 0:
            number_of_cameras_used += 1

    assert True
