import os
import sys
from threading import Event
from time import sleep

# Add 'src' directory to Python path
src_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src/camera_thread/")
)
sys.path.append(src_path)

src_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src/hand_recognizer/")
)
sys.path.append(src_path)

from camera_thread import CameraThreadCV


def test_threads_cv():
    arr = CameraThreadCV.returnCameraIndexes()
    print("Cameras", arr)

    close_threads = Event()
    results, threads = dict(), dict()

    for camera in arr:
        # threads
        results[camera] = list()
        threads[camera] = CameraThreadCV(
            camera, close_threads, results[camera], "models\hand_landmarker.task"
        )
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
