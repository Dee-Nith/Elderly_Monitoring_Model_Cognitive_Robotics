"""
Configuration settings for the Elderly Monitoring System
"""

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20

# Model Settings
MODEL_PATH = "models/action_model.pkl"
ENCODER_PATH = "models/encoders.pkl"
ACTION_ENCODER_PATH = "models/action_encoder.pkl"

# Alert Settings
HIGH_PRIORITY_ACTIONS = ["medication", "emergency_button"]
ALERT_COLOR = (0, 0, 255)  # Red
NORMAL_COLOR = (0, 255, 0)  # Green

# Logging Settings
LOG_EVERY_N_FRAMES = 10
LOG_FILE = "data/elderly_activity_log.csv"
OUTPUT_VIDEO = "data/elderly_monitoring_output.avi"

# Gaze Detection Settings
GAZE_THRESHOLD = 3  # Pixels for left/right detection
IRIS_LANDMARK = 468  # MediaPipe iris landmark index

# Hand Detection Settings
GRIP_DISTANCE_THRESHOLD = 50  # Pixels for grip detection
THUMB_LANDMARK = 4
INDEX_LANDMARK = 8

# Trajectory Settings
TRAJECTORY_HISTORY_SIZE = 10
MOVEMENT_THRESHOLD = 20  # Pixels for movement detection

# Display Settings
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 0)  # Yellow
ALERT_FONT_SCALE = 1.0
ALERT_FONT_THICKNESS = 3
