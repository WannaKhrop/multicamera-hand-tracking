"""
Module describes enumeration that contains description for each landmark.

Author: Ivan Khrop
Date: 26.07.2024
"""

from enum import Enum


class HandLandmark(Enum):
    """
    Contains all constants that describe handlandmarks in mediapipe.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# this list containt all the finger connections using ID-s of landmarks
finger_connections = [
    (HandLandmark.THUMB_CMC.value, HandLandmark.THUMB_MCP.value),
    (HandLandmark.THUMB_MCP.value, HandLandmark.THUMB_IP.value),
    (HandLandmark.THUMB_IP.value, HandLandmark.THUMB_TIP.value),
    (HandLandmark.INDEX_FINGER_MCP.value, HandLandmark.INDEX_FINGER_PIP.value),
    (HandLandmark.INDEX_FINGER_PIP.value, HandLandmark.INDEX_FINGER_DIP.value),
    (HandLandmark.INDEX_FINGER_DIP.value, HandLandmark.INDEX_FINGER_TIP.value),
    (HandLandmark.MIDDLE_FINGER_MCP.value, HandLandmark.MIDDLE_FINGER_PIP.value),
    (HandLandmark.MIDDLE_FINGER_PIP.value, HandLandmark.MIDDLE_FINGER_DIP.value),
    (HandLandmark.MIDDLE_FINGER_DIP.value, HandLandmark.MIDDLE_FINGER_TIP.value),
    (HandLandmark.RING_FINGER_MCP.value, HandLandmark.RING_FINGER_PIP.value),
    (HandLandmark.RING_FINGER_PIP.value, HandLandmark.RING_FINGER_DIP.value),
    (HandLandmark.RING_FINGER_DIP.value, HandLandmark.RING_FINGER_TIP.value),
    (HandLandmark.PINKY_MCP.value, HandLandmark.PINKY_PIP.value),
    (HandLandmark.PINKY_PIP.value, HandLandmark.PINKY_DIP.value),
    (HandLandmark.PINKY_DIP.value, HandLandmark.PINKY_TIP.value),
]

# this list contains indexes that define a palm
palm_landmarks = [
    HandLandmark.WRIST.value,
    HandLandmark.INDEX_FINGER_MCP.value,
    HandLandmark.MIDDLE_FINGER_MCP.value,
    HandLandmark.RING_FINGER_MCP.value,
    HandLandmark.PINKY_MCP.value,
]

# this list contains indexes that define fingers
fingers_landmarks = [
    HandLandmark.THUMB_MCP.value,  # thumb
    HandLandmark.THUMB_IP.value,
    HandLandmark.THUMB_TIP.value,
    HandLandmark.INDEX_FINGER_PIP.value,  # index
    HandLandmark.INDEX_FINGER_DIP.value,
    HandLandmark.INDEX_FINGER_TIP.value,
    HandLandmark.MIDDLE_FINGER_PIP.value,  # middle
    HandLandmark.MIDDLE_FINGER_DIP.value,
    HandLandmark.MIDDLE_FINGER_TIP.value,
    HandLandmark.RING_FINGER_PIP.value,  # ring
    HandLandmark.RING_FINGER_DIP.value,
    HandLandmark.RING_FINGER_TIP.value,
    HandLandmark.PINKY_PIP.value,  # small
    HandLandmark.PINKY_DIP.value,
    HandLandmark.PINKY_TIP.value,
]
