import os
import sys
from threading import Event
from time import sleep

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from camera_thread.rs_thread import CameraThreadRS


def test_threads_rs():
    arr = CameraThreadRS.returnCameraIndexes()

    assert len(arr) > 0, "No cameras are available. Test can not be passed."

    close_threads = Event()
    results, threads = dict(), dict()

    for camera_name, camera_id in arr:
        # threads
        results[camera_id] = list()
        threads[camera_id] = CameraThreadRS(
            camera_name, camera_id, close_threads, results[camera_id]
        )
        threads[camera_id].start()

    # sleep for a while
    sleep(10)
    close_threads.set()

    # check results
    number_of_cameras_used = 0
    for camera_name, camera_id in arr:
        if len(results[camera_id]) > 0:
            number_of_cameras_used += 1

    assert number_of_cameras_used == len(arr), "Not all cameras are used"
