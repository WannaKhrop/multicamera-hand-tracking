"""
Module contains implementation of RealSense Camera class.

Author: Mikula Fabian
Extention: Ivan Khrop
Date: 03.08.2024
"""

import cv2
import pyrealsense2 as rs
import numpy as np

from pathlib import Path
from utils.constants import PATH_TO_TRANSFORMATION, NUMPY_FILE_EXT, SCALE_FACTOR


class camera:
    """
    Class describes RealSense Camera.

    Attributes
    ----------
    pipeline: rs.pipeline
        Camera main structure.
    config: rs.config
        Configuration of a camera.
    device_name: str
        Name of a camera.
    device_id: int
        Unique ID of a camera.
    pipeline_started: bool
        Flag that the camera started taking frames.
    intrinsics_saved: bool
        Flag that intrinsics parameters are saved.
    aling: !!! TODO !!!
        Color alignment.
    depth_frame: !!! TODO !!!
        Frame that contains depth for each pixel.
    color_frame: !!! TODO !!!
        A colored picture taken.
    depth_intrin: !!! TODO !!!
        Intrinsics parameters for depth frame.
    color_intrin: !!! TODO !!!
        Intrinsics parametrers for color frame.
    depth_scale: !!! TODO !!!
        Depth scale from closest to furthest point.
    """

    def __init__(self, device_name: str, device_id: str):
        """
        Initialize a new camera instance.

        Parameters
        ----------
        device_name: str
            Name of a camera
        device_id: str
            Unique ID of a camera
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # internal data
        self.device_name = device_name
        self.device_id = device_id
        self.pipeline_started = False
        self.intrinsics_saved = False

        # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = self.config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.config.enable_device(self.device_id)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

        self.intrinsics = rs.intrinsics()

    def take_picture_and_return_color(self):
        """
        Capture the current frame.

        Returns
        -------
        np.ndarray:
            The taken image.
        """
        # start pipeline
        if not self.pipeline_started:
            self.pipeline.start(self.config)

            # load profile for intrinsics and extrinsics
            active_profile = self.pipeline.get_active_profile()

            # get extrinsics
            self.depth_to_color_extrin = (
                active_profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_extrinsics_to(active_profile.get_stream(rs.stream.color))
            )
            self.color_to_depth_extrin = (
                active_profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_extrinsics_to(active_profile.get_stream(rs.stream.depth))
            )

            # get depth scale
            self.depth_scale = (
                active_profile.get_device().first_depth_sensor().get_depth_scale()
            )
            # take some pictures till quality is good
            for _ in range(1, 100):
                self.pipeline.wait_for_frames()
                self.pipeline_started = True

        # get frames
        frames = self.pipeline.wait_for_frames()

        # align color and depth stream
        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        # get intrinsics
        self.depth_intrin = (
            self.depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )
        self.color_intrin = (
            self.color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )

        imgRGB = np.asanyarray(self.color_frame.get_data()).copy()
        imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)

        return imgRGB

    def get_last_depth_frame(self) -> np.ndarray:
        """
        Get the latest depth frame.

        Returns
        -------
        np.ndarray
        """
        depth_frame = np.asanyarray(self.depth_frame.get_data()).copy()
        return depth_frame

    def get_last_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """
        Get the latest intrinsics parameters.

        Returns
        -------
        rs.pyrealsense2.intrinsics
        """
        # Create a new intrinsics object
        copied_intrinsics = rs.intrinsics()

        # Manually copy all attributes from the original intrinsics object
        copied_intrinsics.width = self.color_intrin.width
        copied_intrinsics.height = self.color_intrin.height
        copied_intrinsics.ppx = self.color_intrin.ppx
        copied_intrinsics.ppy = self.color_intrin.ppy
        copied_intrinsics.fx = self.color_intrin.fx
        copied_intrinsics.fy = self.color_intrin.fy
        copied_intrinsics.model = self.color_intrin.model
        copied_intrinsics.coeffs = (
            self.color_intrin.coeffs.copy()
        )  # Coefficients array should be copied

        return copied_intrinsics

    def get_depth_data_from_pixel(self, px, py):
        # depth_min = 0.11 #meter
        # depth_max = 1.0 #meter

        # depth_point = rs.rs2_project_color_pixel_to_depth_pixel(self.depth_frame.get_data(), self.depth_scale, depth_min, depth_max, self.depth_intrin, self.color_intrin, self.depth_to_color_extrin, self. color_to_depth_extrin, [px, py])

        # need to check the coordinates of x and y
        try:
            depth = self.depth_frame.get_distance(int(px), int(py))
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(
                self.color_intrin, [px, py], depth
            )
            return [dx, dy, dz]
        except Exception:
            return [-1, -1, -1]

    def __del__(self):
        """Destroy an instance of a camera."""
        if self.pipeline_started:
            self.pipeline.stop()

    def stop(self):
        """Stop camera."""
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False

    @classmethod
    def get_camera_coordinates(
        cls,
        x_pixel: int,
        y_pixel: int,
        depth_frame: np.ndarray,
        intrinsics: rs.pyrealsense2.intrinsics,
    ) -> np.ndarray:
        # need to check the coordinates of x and y
        try:
            depth = depth_frame[y_pixel, x_pixel] / SCALE_FACTOR
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(
                intrinsics, [x_pixel, y_pixel], depth
            )
            return np.array([dx, dy, dz])
        except Exception:
            return np.array([-10, -10, -10])

    @classmethod
    def get_coordinates_for_depth(
        cls,
        x_pixel: int,
        y_pixel: int,
        depth: float,
        intrinsics: rs.pyrealsense2.intrinsics,
    ) -> np.ndarray:
        try:
            dx, dy, dz = rs.rs2_deproject_pixel_to_point(
                intrinsics, [x_pixel, y_pixel], depth
            )
            return np.array([dx, dy, dz])
        except Exception:
            return np.array([-10, -10, -10])

    @classmethod
    def get_depth(cls, x_pixel: int, y_pixel: int, depth_frame: np.ndarray) -> float:
        """
        Get camera coordinates for pixel (x, y).

        Parameters
        ----------
        x_pixel: int
            X-pixel.
        y_pixel: int
            Y-pixel.
        depth_frame: np.ndarray
            Depth data from image.

        Returns
        -------
        float
            Depth data for pixel.
        """
        depth = 0.0
        try:
            depth = depth_frame[y_pixel, x_pixel] / SCALE_FACTOR
        except Exception:
            pass

        return depth

    @classmethod
    def get_transformation_matrix(cls, camera_id: str) -> np.ndarray:
        """
        Read transformation matrix for camera.

        Parameters
        ----------
        camera_id: str
            Camera ID that transformation matrix must be read for.

        Returns
        -------
        np.ndarray
            Tranformation matrix from camera coordinates to world coordinates [R|t].
        """
        # read file
        file_path = (
            Path(__file__)
            .parent.parent.parent.joinpath(PATH_TO_TRANSFORMATION)
            .joinpath(camera_id + NUMPY_FILE_EXT)
        )

        try:
            matrix = np.load(file=file_path)
            return matrix
        except FileNotFoundError as error:
            print(f"File not found: {error}")
        except ValueError:
            print(
                f"File at {file_path} is not a valid .npy file or contains unsupported dtype."
            )
        except OSError as e:
            print(f"OS error while reading the file: {e}")
        except EOFError:
            print(f"Unexpected end of file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return np.zeros(1)
