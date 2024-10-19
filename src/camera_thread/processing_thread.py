"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event, Barrier, BrokenBarrierError
import numpy as np
from typing import Any

# other imports
from camera_thread.rs_thread import CameraThreadRS
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
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

    def run(self):
        """Run thread and process results."""
        # untill threads stopped
        while not self.stop_thread.is_set():
            # give time for other threads and wait for data
            try:
                self.data_barrier.wait(timeout=DATA_WAIT_TIME)
            except BrokenBarrierError:
                self.stop_thread.set()
                continue

            # read data
            data = [self.sources[source].get_frame() for source in self.sources]

            # if there is a source with new data
            self.process_sources(self, data)

        # report finish !!!
        print("Fusion thread is stopped")

    @TimeChecker
    def process_sources(self, data: list[tuple[Any, Any, Any, Any, Any]]):
        """Go over all sources, get the latest results and fuse them."""
        # collect data from threads
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
                hand_depths: list[np.ndarray] = list()
                for hand in detected_hands:
                    # extract features
                    rel_depths, depths = convert_to_features(
                        detected_hands[hand], depth_frame=depth_frame
                    )
                    # get min depth bu more than 0.0
                    min_depth = np.min(depths[depths > 1e-3])
                    # get argmin
                    argmin = np.argmin(np.abs(depths - min_depth))
                    # update_relative depths
                    rel_depths = 1.0 + rel_depths - rel_depths[argmin]
                    # save new depths
                    hand_depths.append(rel_depths * min_depth)

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

        # do fusion as all processes are finished
        if len(data) > 0:
            self.merger.make_fusion(self.merger)
