import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from matplotlib import pyplot as plt
from enum import Enum
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class HandLandmark(Enum):
    """
    Contains all constants that describe handlandmarks in mediapipe
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

def process_image(image: np.ndarray):
    """
    Returns results of the detection for an image

    Parameters
    ----------
    image: np.ndarray
        Image that must be processed
    
    Returns
    -------
    detection_results: mediapipe.
        Results of mediapipe
    """

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='D:\Project\multicamera-handtracking\models\hand_landmarker.task'),
        num_hands=2,
        running_mode=VisionRunningMode.IMAGE)

    with HandLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    print(type(results))

    return results

def draw_hand(hand_landmarks: list[NormalizedLandmark], azimuth: int = 10, elevation: int = 10):
    """
    Draws 3D model for the provided list of landmarks

    Parameters
    ----------
    hand_landmarks: list[NormalizedLandmark]
        List of detected handlandmarks to be drawn
    azimuth: int
        Angle to turn the 3D-plot
    elevation:
        Vakue to elevate the 3D-plot
    """

    def _normalize_color(color):
        return tuple(v / 255. for v in color)

    landmark_drawing_spec = solutions.drawing_styles.get_default_hand_landmarks_style()
    connection_drawing_spec = solutions.drawing_styles.get_default_hand_connections_style()

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}

    for idx, landmark in enumerate(hand_landmarks):

        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec[idx].color[::-1]),
            linewidth=landmark_drawing_spec[idx].thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        ax.text(-landmark.z, landmark.x, -landmark.y, str(idx), size=12, zorder=0, color='black')

    num_landmarks = len(hand_landmarks)
    
    # Draws the connections if the start and end landmarks are both visible.
    for connection in solutions.hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        key = (start_idx, end_idx)
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                                f'from landmark #{start_idx} to landmark #{end_idx}.')

        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [plotted_landmarks[start_idx], plotted_landmarks[end_idx]]
        
            ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color=_normalize_color(connection_drawing_spec[key].color[::-1]),
            linewidth=connection_drawing_spec[key].thickness)
    plt.show()

def mplist_to_np_array(detection_results) ->  np.ndarray:
    