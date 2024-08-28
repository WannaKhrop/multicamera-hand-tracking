# import threads
from camera_thread.rs_thread import CameraThreadRS

# in case of saving images
from collections import deque
from time import time

# event to stop threads
from threading import Event
import warnings

# functions to process images
from hand_recognition.hand_recognizer import convert_to_camera_coordinates_holistic
from hand_recognition.hand_recognizer import process_image
from hand_recognition.hand_recognizer import draw_hand_animated
from utils.utils import merge_sorted_lists
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
from utils.constants import TIME_DELTA

"""
# Create a context object. This object owns the handles to all connected realsense devices
my_cam = camera(device_name="Intel RealSense D435", device_id="805312070126")

txt = input("To capture: ")

# frame
frame = my_cam.take_picture_and_return_color()
depth_frame = my_cam.get_last_depth_frame()
intrinsics = my_cam.get_last_intrinsics()
my_cam.stop()

# define transformer
transformer = CoordinateTransformer()

# save image
image = Image.fromarray(frame)
image.save("output_image.png")

# get landmarks
mp_results = process_image(frame, holistic=True)
draw_landmarks_holistics(frame, mp_results.right_hand_landmarks)
draw_landmarks_holistics(frame, mp_results.left_hand_landmarks)

# save image
image = Image.fromarray(frame)
image.save("output_image.png")

detected_hands = convert_to_camera_coordinates_holistic(mp_results, depth_frame, intrinsics)

axis = ['x', 'y', 'z']
for hand in detected_hands:
    # world coords
    detected_hands[hand][axis] = transformer.camera_to_world(camera_id="805312070126", 
                                                             points=detected_hands[hand][axis].values)

print(detected_hands)
"""


def main():
    # read available cameras
    available_cameras = CameraThreadRS.returnCameraIndexes()
    # check cameras
    assert (
        len(available_cameras) > 0
    ), "No cameras are available. Test can not be passed."

    # define events and data
    close_threads = Event()
    results = dict()

    for camera_name, camera_id in available_cameras:
        # threads
        results[camera_id] = deque()
        thread = CameraThreadRS(
            camera_name, camera_id, close_threads, results[camera_id]
        )
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

    # gather results from all threads and store them in one list
    all_frames = list()
    start_time = time()
    for camera_id in results:
        all_frames = merge_sorted_lists(all_frames, results[camera_id])
    print(f"Merging all results = {round(time() - start_time, 3)} sec.")

    # define transformer
    transformer = CoordinateTransformer()

    # process each frame and save those ones that have detected hands
    detected_results = list()
    start_time = time()
    for timestamp, camera_id, color_frame, depth_frame, intrinsics in all_frames:
        # process image and get information
        mp_results = process_image(color_frame, holistic=True)
        # draw_landmarks_holistics(color_frame, mp_results.right_hand_landmarks)
        # draw_landmarks_holistics(color_frame, mp_results.left_hand_landmarks)

        # convert coords
        detected_hands = convert_to_camera_coordinates_holistic(
            mp_results, depth_frame, intrinsics
        )

        # check if it's empty then MediaPipe has not found hand on this frame
        if len(detected_hands) == 0:
            continue

        # assign convert to world coordinates and assign visibility to each frame
        axis = ["x", "y", "z"]
        for hand in detected_hands:
            # world coords
            detected_hands[hand][axis] = transformer.camera_to_world(
                camera_id=camera_id, points=detected_hands[hand][axis].values
            )

        # save detected results
        detected_results.append((timestamp, camera_id, detected_hands))
    print(f"Convertion of all results = {round(time() - start_time, 3)} sec.")

    # process close timestamps in time
    data_merger = DataMerger(time_delta=TIME_DELTA)
    start_time = time()
    # process each available frame
    for frame in detected_results:
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
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="google.protobuf.symbol_database"
    )
    main()
