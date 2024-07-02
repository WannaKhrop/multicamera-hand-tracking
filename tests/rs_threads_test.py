import sys
import os

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/camera_thread'))
sys.path.append(src_path)
    

from camera_thread import CameraThreadRS
from threading import Event

def main():

    arr = CameraThreadRS.returnCameraIndexes()
    print("Cameras", arr)

    close_threads = Event()
    results, threads = dict(), dict()

    for camera_name, camera_id in arr:
        # threads
        results[camera_id] = list()
        threads[camera_id] = CameraThreadRS(camera_name, camera_id, close_threads, results[camera_id])
        threads[camera_id].start()

    while True:
        data = input("Type 'close' to exit: ")
        if data == 'close':
            close_threads.set()
            break

if __name__ == '__main__':
    main()