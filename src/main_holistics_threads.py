# import threads
from camera_thread.rs_thread import CameraThreadRS

# in case of saving images
from collections import deque
from time import time, sleep
import pandas as pd

# event to stop threads
from threading import Event
import warnings

# functions to process images
from hand_recognition.hand_recognizer import draw_hand_animated_plotly, draw_hand_animated
from utils.utils import merge_sorted_lists
from utils.fusion import DataMerger
from utils.constants import TIME_DELTA


def main():
    # read available cameras
    available_cameras = CameraThreadRS.returnCameraIndexes()
    # check cameras
    assert (
        len(available_cameras) > 0
    ), "No cameras are available. Test can not be passed."

    # define events and data
    close_threads = Event()
    results: dict[str, deque[tuple[int, str, dict[str, pd.DataFrame]]]] = dict()
    threads: dict[str, CameraThreadRS] = dict()

    for camera_name, camera_id in available_cameras:
        # threads
        results[camera_id] = deque()
        thread = CameraThreadRS(
            camera_name,
            camera_id,
            close_threads,
            results[camera_id],
            use_holistics=True,
        )

        threads[camera_id] = thread
        thread.start()

    # !!! TODO !!!
    # add event to start threads
    print("Threads started")

    # wait till the end
    while True:
        stop_threds = input("Input 'stop' to interrupt all thread: ")
        if stop_threds == "stop":
            close_threads.set()
            break

    # wait all threads
    while True:
        stopped = [threads[camera_id].is_alive() for camera_id in threads]
        if any(stopped):
            sleep(1.0)
            continue
        else:
            break

    # gather results from all threads and store them in one list
    all_frames: list[tuple[int, str, dict[str, pd.DataFrame]]] = list()
    start_time = time()
    for camera_id in results:
        all_frames = list(merge_sorted_lists(all_frames, results[camera_id]))
    print(f"Merging all results = {round(time() - start_time, 3)} sec.")

    # process close timestamps in time
    data_merger = DataMerger(time_delta=TIME_DELTA)
    start_time = time()
    # process each available frame
    for frame in all_frames:
        print(frame[0], frame[1])
        data_merger.add_time_frame(*frame)
    print(f"Fusion = {round(time() - start_time, 3)} sec.")

    # get results
    final_result_right, final_result_left = list(), list()
    for timestamp, hands in data_merger.fusion_results:
        if "Right" in hands:
            final_result_right.append((timestamp, hands["Right"]))
        if "Left" in hands:
            final_result_left.append((timestamp, hands["Left"]))

    draw_hand_animated(final_result_right)


if __name__ == "__main__":
    # ignore the following warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
