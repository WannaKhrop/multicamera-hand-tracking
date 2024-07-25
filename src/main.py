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

import cv2

from hand_recognizer import hand_recognizer

video = cv2.VideoCapture(0)

ret, frame = video.read()

video.release()

results = hand_recognizer.process_image(hand_recognizer.create_landmarker(), frame)

"""
# get hand world landmarks
world_landmarks = hand_recognizer.to_numpy_ndarray(results.hand_world_landmarks[0])

# get normalized landmarks
landmarks = hand_recognizer.to_numpy_ndarray(results_of_mediapipe.hand_landmarks[0])

# get the closest point to the camera according to z-axis
closest_point_idx = hand_recognizer.HandLandmark(np.argmin(landmarks[:, 2]))
closest_point = hand_recognizer.extract_camera_coordinates(results_of_mediapipe.hand_landmark[0][closest_point_idx],
                                                          camera)

# make the closest point a new center of coordinates
hand_with_new_origin = hand_recognizer.change_origin(closest_point_idx, world_landmarks)

# add the real world coordinates to the camera coordinates
# save result for hand
hands[name] = closest_point + hand_with_new_origin
"""

hand_recognizer.draw_hand(results.hand_world_landmarks[0])

print(results.hand_world_landmarks[0])
