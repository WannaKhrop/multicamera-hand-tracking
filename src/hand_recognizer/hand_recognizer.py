import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from matplotlib import pyplot as plt

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def process_image(image):

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='D:\Project\multicamera-handtracking\models\hand_landmarker.task'),
        num_hands=2,
        running_mode=VisionRunningMode.IMAGE)

    with HandLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    return results

def draw_hand(hand_landmarks, azimuth=10, elevation=10):

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