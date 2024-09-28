"""
Module gathers information from different threads and makes fusion async.

Author: Ivan Khrop
Data: 08.08.2024
"""
# basic imports
from threading import Thread, Event
from time import sleep

# other imports
from camera_thread.rs_thread import CameraThreadRS
from utils.coordinate_transformer import CoordinateTransformer
from utils.fusion import DataMerger
from hand_recognition.hand_recognizer import convert_to_camera_coordinates_holistic


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
            for source in self.sources:
                # get frame
                timestamp, _, mp_results, depth_frame, intrinsics = self.sources[
                    source
                ].get_frame()

                # check if we have something
                if timestamp is not None:
                    # detect hands
                    detected_hands = convert_to_camera_coordinates_holistic(
                        mp_results, depth_frame, intrinsics
                    )

                    # if something is detected
                    if len(detected_hands) > 0:
                        # assign convert to world coordinates and assign visibility to each frame
                        axes = ["x", "y", "z"]
                        for hand in detected_hands:
                            # world coords
                            detected_hands[hand].loc[
                                :, axes
                            ] = self.transformer.camera_to_world(
                                camera_id=source,
                                points=detected_hands[hand].loc[:, axes].values,
                            )

                        # make fusion
                        self.merger.add_time_frame(timestamp, source, detected_hands)

            # sleep a bit
            sleep(0.005)
