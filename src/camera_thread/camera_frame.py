"""
This module contains a data structure for on captured frame of a camera.

Author: Ivan Khrop
Date: 13.01.2025
"""
import pandas as pd
import pyrealsense2 as rs

# Map distortion enum to a string
forward_distortion_mapping = {
    rs.distortion.none: "none",
    rs.distortion.modified_brown_conrady: "modified_brown_conrady",
    rs.distortion.inverse_brown_conrady: "inverse_brown_conrady",
    rs.distortion.brown_conrady: "brown_conrady",
    rs.distortion.kannala_brandt4: "kannala_brandt4",
}
# Map distortion model string back to rs.distortion
backward_distortion_mapping = {
    "none": rs.distortion.none,
    "modified_brown_conrady": rs.distortion.modified_brown_conrady,
    "inverse_brown_conrady": rs.distortion.inverse_brown_conrady,
    "brown_conrady": rs.distortion.brown_conrady,
    "kannala_brandt4": rs.distortion.kannala_brandt4,
}


class CameraFrame:
    """
    Class that describes a frame of a camera.

    Attributes
    ----------
    timestamp: int
        Time when the frame was captured
    camera_id: str
        ID of a camera that captured the frame
    landmarks: dict[str, pd.DataFrame]
        Landmarks of a left and right hands
    intrinsics: rs.pyrealsense2.intrinsics | None
        Intrinsics of a camera
    """

    def __init__(
        self,
        timestamp: int,
        camera_id: str,
        landmarks: dict[str, pd.DataFrame],
        intrinsics: rs.pyrealsense2.intrinsics | None,
    ):
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.landmarks = landmarks
        self.intrinsics = intrinsics

    def to_dict(self) -> dict:
        """
        Convert the frame to a dictionary.

        Returns
        -------
        dict
            Dictionary with the frame attributes
        """
        # get data regarding the predicted hands
        coordinates = {
            key: self.landmarks[key].to_dict() for key in self.landmarks.keys()
        }
        # retrieve the intrinsics of the camera
        if self.intrinsics is not None:
            intrinsics = {
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "ppx": self.intrinsics.ppx,
                "ppy": self.intrinsics.ppy,
                "model": forward_distortion_mapping.get(
                    self.intrinsics.model, "unknown"
                ),
                "coeffs": self.intrinsics.coeffs,
            }
        else:
            intrinsics = None
        # depth data

        return {
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "landmarks": coordinates,
            "intrinsics": intrinsics,
        }

    def as_tuple(
        self,
    ) -> tuple[int, str, dict[str, pd.DataFrame], rs.pyrealsense2.intrinsics | None]:
        """
        Convert the frame to a tuple.

        Returns
        -------
        tuple
            Tuple with the frame attributes in the following order: timestamp, camera_id, landmarks, intrinsics
        """
        return (
            self.timestamp,
            self.camera_id,
            self.landmarks,
            self.intrinsics,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "CameraFrame":
        """
        Create a new instance of CameraFrame from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary with the frame attributes

        Returns
        -------
        CameraFrame
            New instance of CameraFrame
        """
        # get data regarding the predicted hands
        landmarks = {
            key: pd.DataFrame(data["landmarks"][key])
            for key in data["landmarks"].keys()
        }
        for key in landmarks:
            landmarks[key].index = landmarks[key].index.map(
                lambda landmark_id: int(landmark_id)
            )
        # retrieve the intrinsics of the camera
        if data["intrinsics"] is not None:
            intrinsics = rs.intrinsics()
            intrinsics.width = data["intrinsics"]["width"]
            intrinsics.height = data["intrinsics"]["height"]
            intrinsics.fx = data["intrinsics"]["fx"]
            intrinsics.fy = data["intrinsics"]["fy"]
            intrinsics.ppx = data["intrinsics"]["ppx"]
            intrinsics.ppy = data["intrinsics"]["ppy"]
            intrinsics.model = backward_distortion_mapping.get(
                data["intrinsics"]["model"], rs.distortion.none
            )
            intrinsics.coeffs = data["intrinsics"]["coeffs"]
        else:
            intrinsics = None

        return cls(
            data["timestamp"],
            data["camera_id"],
            landmarks,
            intrinsics,
        )

    def copy(self) -> "CameraFrame":
        """
        Create a copy of the frame.

        Returns
        -------
        CameraFrame
            Copy of the frame
        """
        return CameraFrame(
            self.timestamp,
            self.camera_id,
            {key: self.landmarks[key].copy() for key in self.landmarks.keys()},
            self.intrinsics,
        )
