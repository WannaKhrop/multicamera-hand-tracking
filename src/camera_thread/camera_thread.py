from threading import Thread, Event
from time import time

import cv2
import numpy as np

# realsense camera
from camera import camera
import pyrealsense2 as rs

class CameraThreadCV(Thread):

    def __init__(self, camera_id: int, 
                 close_event: Event, 
                 target: list[tuple[int, np.array]], 
                 stream: bool = False):
        
        Thread.__init__(self)
        self.camera_id = camera_id
        self.close_event = close_event
        self.capture_target = target
        self.stream = stream

    def get_name(self) -> str:
        return 'Camera #{}'.format(self.camera_id)

    def run(self):

        video = cv2.VideoCapture(self.camera_id)

        # for testing purposes
        if self.stream:
            cv2.namedWindow(self.get_name())

        while video.isOpened():

            ret, frame = video.read()
            time_stamp = int(time() * 1000)

            self.capture_target.append((time_stamp, frame))

            # for testing purposes
            if self.stream:
                cv2.imshow(self.get_name(), frame)

            if self.close_event.is_set() or not ret:
                break
        
        video.release()

        # for testing purposes
        if self.stream:
            cv2.destroyWindow(self.get_name())

    @classmethod
    def returnCameraIndexes(cls) -> list[int]:

        arr = []

        for index in range(10):
            cap = cv2.VideoCapture(index)
            ret, _ = cap.read()
            if ret:
                arr.append(index)
                cap.release()

        return arr

class CameraThreadRS(Thread):

    def __init__(self, 
                 camera_name: str, 
                 camera_id: int,
                 close_event: Event, 
                 target: list[tuple[int, np.array]]):
        
        Thread.__init__(self)
        self.camera = camera(camera_name, camera_id)
        self.close_event = close_event
        self.capture_target = target

    def get_name(self) -> str:
        return 'Camera #{}'.format(self.camera.device_id)

    def run(self):

        while True:

            frame = self.camera.take_picture_and_return_color()
            time_stamp = int(time() * 1000)

            self.capture_target.append((time_stamp, frame))

            if self.close_event.is_set():
                break
    
    @classmethod
    def returnCameraIndexes(cls) -> list[tuple[str, int]]:

        arr = []

        context = rs.context()
        for device in context.devices:
            arr.append((device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number)))

        return arr