"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event
import numpy as np
import pyrealsense2 as rs
from time import sleep

# other imports
from camera_thread.rs_thread import CameraThreadRS
from hand_recognition.HolisticLandmarker import HolisticLandmarker
from utils.coordinate_transformer import CoordinateTransformer
from hand_recognition.hand_recognizer import convert_to_camera_coordinates_holistic
from utils.fusion import DataMerger


class FusionThread(Thread):
    """
    Class that describes thread that makes fusion and saves the results.

    Attributes
    ----------
    stop_thread: Event
        Event to stop all threads.
    fusion_results: list[tuple[int, pd.DataFrame]]
        Resulting world coordinates of all landmarks at timestamp.
    data_source: dict[int, deque[tuple[int, np.array, np.array, rs.pyrealsense2.intrinsics]]]
        Deques where data come from.
    """

    stop_thread: Event
    sources: dict[str, CameraThreadRS]
    indexes: dict[str, int] = dict()
    landmarkers: dict[str, HolisticLandmarker] = dict()
    merger: DataMerger
    transformer: CoordinateTransformer = CoordinateTransformer()

    def __init__(
        self, stop_thread: Event, sources: dict[str, CameraThreadRS], merger: DataMerger
    ):
        """Initialize a new instance of Thread."""
        Thread.__init__(self)
        self.stop_thread = stop_thread
        self.sources = sources
        self.merger = merger

        for source in self.sources:
            self.indexes[source] = 0
            self.landmarkers[source] = HolisticLandmarker()

    def get_next_source(self) -> str:
        """Check if sources do not have any data to process and can not generate any data mode."""

        earliest = 2**64
        next_source = ""
        for source in self.indexes:
            if len(self.sources[source].frames) > self.indexes[source]:
                idx = self.indexes[source]
                if self.sources[source].frames[idx][0] < earliest:
                    next_source = self.sources[source].frames[idx][1]
                    earliest = self.sources[source].frames[idx][0]

        return next_source

    def pick_next_frame(
        self, source: str
    ) -> tuple[int, str, np.ndarray, np.ndarray, rs.pyrealsense2.intrinsics]:
        """Get next frame and move index."""
        # always take the latest frame because we can not process all of them
        # processing one frame takes a lot of time, so we process only the last one
        next_frame = self.sources[source].get_frame(idx=-1)
        self.indexes[source] += 1

        return next_frame

    def run(self):
        """Run thread and process results."""
        # untill threads stopped
        while not self.stop_thread.is_set() or self.get_next_source():
            # get frame
            source = self.get_next_source()

            # if there is a source with new data
            if source in self.sources:
                # get frame
                (
                    timestamp,
                    _,
                    color_frame,
                    depth_frame,
                    intrinsics,
                ) = self.pick_next_frame(source)

                # process frame with mediapipe + annotate image
                mp_results = self.landmarkers[source].process_image(color_frame)
                # draw_landmarks_holistics(color_frame, mp_results.left_hand_landmarks)
                # draw_landmarks_holistics(color_frame, mp_results.right_hand_landmarks)

                # process hands
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
                    detected_hands[hand][axis] = self.transformer.camera_to_world(
                        camera_id=source, points=detected_hands[hand][axis].values
                    )

                # make fusion
                self.merger.add_time_frame(timestamp, source, detected_hands)

            # sleep a bit
            sleep(0.001)
