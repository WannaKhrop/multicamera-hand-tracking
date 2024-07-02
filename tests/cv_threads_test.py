import sys
import os

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/camera_thread'))
sys.path.append(src_path)
    

from camera_thread import CameraThreadCV
from threading import Event

def main():

    arr = CameraThreadCV.returnCameraIndexes()
    print("Cameras", arr)

    close_threads = Event()
    results, threads = dict(), dict()

    for camera in arr:
        # threads
        results[camera] = list()
        threads[camera] = CameraThreadCV(camera, close_threads, results[camera], stream=True)
        threads[camera].start()

    while True:
        data = input("Type 'close' to exit: ")
        if data == 'close':
            close_threads.set()
            break

if __name__ == '__main__':
    main()