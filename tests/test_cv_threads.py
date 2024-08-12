import os
import sys
from threading import Event
from time import sleep

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/"))
sys.path.append(src_path)

from camera_thread.cv_thread import CameraThreadCV


def test_threads_cv():
    arr = CameraThreadCV.returnCameraIndexes()

    assert len(arr) > 0, "No cameras are available. Test can not be passed."

    close_threads = Event()
    results, threads = dict(), dict()

    for camera in arr:
        # threads
        results[camera] = list()
        threads[camera] = CameraThreadCV(camera, close_threads, results[camera])
        threads[camera].start()

    # sleep for a while
    sleep(20)
    close_threads.set()

    # check results
    number_of_cameras_used = 0
    for camera in arr:
        if len(results[camera]) > 0:
            number_of_cameras_used += 1

    assert number_of_cameras_used == len(arr), "Not all cameras are used"
