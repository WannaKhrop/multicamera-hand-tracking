"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event
import pandas as pd
from time import sleep

# other imports
from camera_thread.rs_thread import CameraThreadRS
from utils.coordinate_transformer import CoordinateTransformer
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

    def pick_next_frame(
        self, source: str
    ) -> tuple[int, str, dict[str, pd.DataFrame]] | tuple[None, None, None]:
        """Get next frame and move index."""
        # always take the latest frame because we can not process all of them
        # processing one frame takes a lot of time, so we process only the last one
        return self.sources[source].get_frame(idx=-1)

    def run(self):
        """Run thread and process results."""
        # untill threads stopped
        while not self.stop_thread.is_set():
            # if there is a source with new data
            for source in self.sources:
                # get frame
                timestamp, _, detected_hands = self.sources[source].get_frame(idx=-1)

                # check if we have something
                if timestamp is None or detected_hands is None:
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
            sleep(0.01)
