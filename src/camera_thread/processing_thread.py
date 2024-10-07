"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event
from time import sleep
import numpy as np

# other imports
from camera_thread.rs_thread import CameraThreadRS
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
from utils.mediapipe_world_model import MedapipeWorldTransformer
from hand_recognition.hand_recognizer import convert_to_features, retrieve_from_depths
from utils.utils import TimeChecker


class FusionThread(Thread):
    """
    Class that describes thread that makes fusion and saves the results.

    Attributes
    ----------
    stop_thread: Event
        Event to stop all threads.
    data_source: dict[int, deque[tuple[int, np.array, np.array, rs.pyrealsense2.intrinsics]]]
        Deques where data come from.
    merger: DataMerger
        Structure to merge frames from several cameras.
    """

    stop_thread: Event
    sources: dict[str, CameraThreadRS]
    merger: DataMerger
    transformer: CoordinateTransformer = CoordinateTransformer()
    ml_detector: MedapipeWorldTransformer = MedapipeWorldTransformer()

    def __init__(
        self, stop_thread: Event, sources: dict[str, CameraThreadRS], merger: DataMerger
    ):
        """Initialize a new instance of Thread."""
        Thread.__init__(self)
        self.stop_thread = stop_thread
        self.sources = sources
        self.merger = merger

    def run(self):
        """Run thread and process results."""
        # untill threads stopped
        while not self.stop_thread.is_set():
            # if there is a source with new data
            self.process_sources(self)
            # sleep a bit
            sleep(0.02)

    @TimeChecker
    def process_sources(self):
        """Go over all sources, get the latest results and fuse them."""
        for source in self.sources:
            # get frame
            timestamp, _, detected_hands, depth_frame, intrinsics = self.sources[
                source
            ].get_frame()

            # if no results, then just next source
            if (
                timestamp is None
                or detected_hands is None
                or depth_frame is None
                or intrinsics is None
            ):
                continue

            if len(detected_hands) > 0:
                # process each hand
                features = np.empty(shape=(0, 42))
                for hand in detected_hands:
                    # extract features
                    features_hand = convert_to_features(
                        detected_hands[hand], depth_frame=depth_frame
                    ).reshape(1, -1)
                    features = np.vstack([features, features_hand])

                # predict real depths using ml
                hand_depths = self.ml_detector(self.ml_detector, features=features)

                # convert to camera and then to world
                axes = ["x", "y", "z"]
                for hand, depths in zip(detected_hands, hand_depths):
                    # camera coords
                    retrieve_from_depths(
                        landmarks=detected_hands[hand],
                        depths=depths,
                        intrinsics=intrinsics,
                    )
                    # world coords
                    detected_hands[hand].loc[
                        :, axes
                    ] = self.transformer.camera_to_world(
                        camera_id=source,
                        points=detected_hands[hand].loc[:, axes].values,
                    )

                # make fusion
                self.merger.add_time_frame(timestamp, source, detected_hands)
