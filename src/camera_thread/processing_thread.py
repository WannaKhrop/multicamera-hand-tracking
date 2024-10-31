"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event, Barrier, BrokenBarrierError
import numpy as np

# other imports
from camera_thread.rs_thread import CameraThreadRS
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
from utils.mediapipe_world_model import MedapipeWorldTransformer
from hand_recognition.hand_recognizer import convert_to_features, retrieve_from_depths
from utils.utils import TimeChecker
from utils.constants import DATA_WAIT_TIME
from utils.geometry import assign_visibility


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
    ml_detectors: dict[str, MedapipeWorldTransformer]
    data_barrier: Barrier

    def __init__(
        self,
        stop_thread: Event,
        sources: dict[str, CameraThreadRS],
        merger: DataMerger,
        data_barrier: Barrier,
    ):
        """Initialize a new instance of Thread."""
        Thread.__init__(self)
        self.stop_thread = stop_thread
        self.sources = sources
        self.merger = merger
        self.data_barrier = data_barrier

        # create ML Detectors
        self.ml_detectors = dict()
        for camera_id in sources:
            self.ml_detectors[camera_id] = MedapipeWorldTransformer(camera_id=camera_id)

    def run(self):
        """Run thread and process results."""
        # untill threads stopped
        while not self.stop_thread.is_set():
            # give time for other threads and wait for data
            try:
                self.data_barrier.wait(timeout=DATA_WAIT_TIME)
            except BrokenBarrierError:
                print(
                    "Data-Barrier is broken, proceeding without synchronization is impossible."
                )
                self.stop_thread.set()
                break
            # if there is a source with new data
            self.process_sources(self)

        # write a report
        # self.merger.fluctuation_report()

    @TimeChecker
    def process_sources(self):
        """Go over all sources, get the latest results and fuse them."""
        # collect data from threads
        data = [self.sources[source].get_frame() for source in self.sources]

        for timestamp, source, detected_hands, depth_frame, intrinsics in data:
            # if no results, then just next source
            if (
                timestamp is None
                or source is None
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
                hand_depths = self.ml_detectors[source](
                    self.ml_detectors[source], features=features
                )

                # convert to camera and then to world
                axes = ["x", "y", "z"]
                for hand, depths in zip(detected_hands, hand_depths):
                    # camera coords
                    retrieve_from_depths(
                        landmarks=detected_hands[hand],
                        depths=depths,
                        intrinsics=intrinsics,
                    )

                    # assign visibility
                    assign_visibility(detected_hands[hand])

                    # world coords
                    detected_hands[hand].loc[
                        :, axes
                    ] = self.transformer.camera_to_world(
                        camera_id=source,
                        points=detected_hands[hand].loc[:, axes].values,
                    )

                    # make fusion
                    self.merger.add_time_frame(timestamp, source, detected_hands)

        # do fusion
        if len(data) > 0:
            self.merger.make_fusion(self.merger)
