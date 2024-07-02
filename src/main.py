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

import pyrealsense2 as rs
import cv2

from camera_thread.camera import camera
from hand_recognizer.hand_recognizer import process_image, draw_hand

from mediapipe.tasks.python.components.containers import NormalizedLandmark

# Create a context object. This object owns the handles to all connected realsense devices
context = rs.context()

print("Available devices:")
for i, device in enumerate(context.devices):
    camera_name = device.get_info(rs.camera_info.name)
    camera_id = device.get_info(rs.camera_info.serial_number)
    print(f"Device {i}: {device.get_info(rs.camera_info.name)}, Serial Number: {device.get_info(rs.camera_info.serial_number)}")

my_cam = camera(camera_name, camera_id)

input('Input to take a picture:')

color_image = my_cam.take_picture_and_return_color()

# Display the image
cv2.imwrite('fingers.jpg', color_image)

results = process_image(color_image)

for hand_landmarks in results.hand_landmarks:
  
    camera_landmarks = []

    for landmark in hand_landmarks:
        x_pixel = landmark.x * 1920
        y_pixel = landmark.y * 1080

        x, y, z = tuple(my_cam.get_depth_data_from_pixel(x_pixel, y_pixel))
        camera_landmarks.append(NormalizedLandmark(x=x, y=y, z=z))

    mp_landmarks = []

    for landmark in hand_landmarks:
        mp_landmarks.append(landmark)

    draw_hand(camera_landmarks, azimuth=6)