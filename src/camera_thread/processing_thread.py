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
from utils.fusion import DataMerger, CameraFrame
from hand_recognition.hand_recognizer import retrieve_from_depths
from utils.utils import TimeChecker
from utils.constants import DATA_WAIT_TIME
from utils.geometry import assign_visibility
from utils.mediapipe_world_model import MedapipeWorldTransformer


class FusionThread(Thread):
    """
    Class that describes thread that makes fusion and saves the results.

    Attributes
    ----------
    stop_thread: Event
        Event to stop all threads.
    sources: dict[str, CameraThreadRS]
        All sources of data.
    merger: DataMerger
        Object that merges all data.
    transformer: CoordinateTransformer
        Object that transforms coordinates to world coordinates.
    ml_detectors: dict[str, MedapipeWorldTransformer]
        All ML-detectors for each camera.
    data_barrier: Barrier
        Barrier to synchronize all threads.
    test_mode: bool
        Flag to save all merging results to the file.
    """

    stop_thread: Event
    sources: dict[str, CameraThreadRS]
    merger: DataMerger
    transformer: CoordinateTransformer = CoordinateTransformer()
    ml_detectors: dict[str, MedapipeWorldTransformer]
    data_barrier: Barrier
    test_mode: bool = False

    def __init__(
        self,
        stop_thread: Event,
        sources: dict[str, CameraThreadRS],
        merger: DataMerger,
        data_barrier: Barrier,
        test_mode: bool = False,
    ):
        """Initialize a new instance of Thread."""
        Thread.__init__(self)
        self.stop_thread = stop_thread
        self.sources = sources
        self.merger = merger
        self.data_barrier = data_barrier
        self.test_mode = test_mode

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
        if self.test_mode:
            self.write_logs()

        # report finish !!!
        print("Processing Thread is stopped")

    @TimeChecker
    def process_sources(self):
        """Go over all sources, get the latest results and fuse them."""
        # collect data from threads
        frames = [self.sources[source].get_frame() for source in self.sources]

        for frame in frames:
            # get data from frame
            if frame is not None:
                timestamp, source, detected_hands, intrinsics = frame.as_tuple()
            else:
                continue

            # process each hand
            features = np.empty(shape=(0, 42))
            for hand in detected_hands:
                # extract features
                features_hand = np.hstack(
                    [
                        detected_hands[hand].z.values.copy(),
                        detected_hands[hand].depth.values.copy(),
                    ]
                )
                features = np.vstack([features, features_hand])
                # drop depth column as we do not need it anymore
                detected_hands[hand].drop(columns=["depth"], inplace=True)

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
                detected_hands[hand].loc[:, axes] = self.transformer.camera_to_world(
                    camera_id=source,
                    points=detected_hands[hand].loc[:, axes].values,
                )

            # make fusion
            camera_frame = CameraFrame(
                timestamp=timestamp,
                camera_id=source,
                landmarks=detected_hands,
                intrinsics=None,
            )
            self.merger.add_time_frame(camera_frame)

        # do fusion
        if len(frames) > 0:
            self.merger.make_fusion(self.merger)

    def write_logs(self):
        """Write logs to the file."""
        self.merger.write_logs()
