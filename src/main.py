"""
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    
    # Display the image.
    cv2.imshow('Mediapipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
"""
# import threads
from camera_thread.camera import camera
from camera_thread.rs_thread import CameraThreadRS

# in case of saving images
from PIL import Image
from collections import deque
from time import time

# event to stop threads
from threading import Event

# functions to process images
from hand_recognition.hand_recognizer import convert_to_camera_coordinates, hand_to_df
from utils.utils import merge_sorted_lists
from utils.geometry import assign_visability
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
from utils.constants import TIME_DELTA

# Create a context object. This object owns the handles to all connected realsense devices
my_cam = camera(device_name="Intel RealSense D435", device_id="805312070126")

txt = input("To capture: ")

# frame
frame = my_cam.take_picture_and_return_color()
depth_frame = my_cam.get_last_depth_frame()
intrinsics = my_cam.get_last_intrinsics()

# save image
image = Image.fromarray(frame)
image.save("output_image.png")

# define transformer
transformer = CoordinateTransformer()

# get landmarks
detected_hands = convert_to_camera_coordinates(frame, depth_frame, intrinsics)
print(detected_hands)

# assign convert to world coordinates and assign visibility to each frame
for hand in detected_hands:
    # world coords
    world_coords = transformer.camera_to_world(
        camera_id="805312070126", points=detected_hands[hand]
    )
    # convert to pd.DataFrame
    detected_hands[hand] = hand_to_df(detected_hands[hand])
    # assign visibility
    assign_visability(df_landmarks=detected_hands[hand])
    # save results
    world_coords["visibility"] = detected_hands[hand].loc[
        world_coords.index, "visibility"
    ]
    detected_hands[hand] = world_coords


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
    detected_results = []
    start_time = time()
    for timestamp, camera_id, color_frame, depth_frame, intrinsics in all_frames:
        # process image and get information
        detected_hands = convert_to_camera_coordinates(
            color_frame, depth_frame, intrinsics
        )

        # check if it's empty then MediaPipe has not found hand on this frame
        if len(detected_hands) == 0:
            continue

        # assign convert to world coordinates and assign visibility to each frame
        for hand in detected_hands:
            # world coords
            world_coords = transformer.camera_to_world(
                camera_id=camera_id, points=detected_hands[hand]
            )
            # convert to pd.DataFrame
            detected_hands[hand] = hand_to_df(detected_hands[hand])
            # assign visibility
            assign_visability(df_landmarks=detected_hands[hand])
            # save results
            world_coords["visibility"] = detected_hands[hand].loc[
                world_coords.index, "visibility"
            ]
            detected_hands[hand] = world_coords

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
    # final_result = data_merger.fusion_results
