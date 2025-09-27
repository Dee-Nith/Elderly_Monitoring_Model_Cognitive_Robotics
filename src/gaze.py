import numpy as np

# Store center reference globally for adaptive calibration
iris_center_reference = None

def get_gaze_direction(landmarks, w, h):
    """
    Estimate the horizontal gaze direction of an elderly user from facial landmarks.

    Parameters:
    - landmarks: list of Mediapipe facial landmarks
    - w, h: width and height of the frame

    Returns:
    - str: one of ['left', 'center', 'right'] based on relative iris position
    """
    global iris_center_reference

    # Get horizontal positions of eye corners and iris
    x_left = int(landmarks[33].x * w)     # Outer left eye corner
    x_right = int(landmarks[133].x * w)   # Outer right eye corner
    x_iris = int(landmarks[468].x * w)    # Center of iris

    # Midpoint between eye corners
    eye_center_x = (x_left + x_right) // 2
    offset = x_iris - eye_center_x

    # Set neutral reference on first frame
    if iris_center_reference is None:
        iris_center_reference = x_iris
        return "center"

    delta = x_iris - iris_center_reference

    # Tune thresholds based on typical elderly eye movement range
    if delta < -3:
        return "left"
    elif delta > 3:
        return "right"
    else:
        return "center"

# Future improvement:
# Consider vertical gaze estimation for "looking down" (fall risk) or "up" (reaching high).
