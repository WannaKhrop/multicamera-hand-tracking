from threading import Thread, Event
import cv2
from time import time

def returnCameraIndexes():

    arr = []

    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()

    return arr

class CameraThread(Thread):

    def __init__(self, camera_id: int, close_event: Event, target: list[tuple[int, cv2.Image]]):
        super().__init__(self)
        self.id = camera_id
        self.captured = None
        self.to_close = close_event
        self.capture_target = target

    def get_name(self):
        return 'Camera #{}'.format(self.id)

    def run(self):

        video = cv2.VideoCapture(self.id)

        while video.isOpened():

            ret, frame = video.read()
            time_stamp = int(time() * 1000)

            if self.close_event.is_set() or not ret:
                break

            self.capture_target.append((time_stamp, frame))
        
        video.release()