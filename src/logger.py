import csv
import os
from datetime import datetime

class ActionLogger:
    def __init__(self, filename="elderly_activity_log.csv"):
        """
        Logger for elderly care monitoring.
        Stores prediction data and alerts in a CSV file.
        """
        self.filename = filename
        self.fields = ["timestamp", "frame", "gaze", "hand", "trajectory", "predicted_action", "alert_flag"]

        # Create log file and header if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, frame_index, gaze, hand, trajectory, action):
        """
        Logs a single frame of prediction data.

        Parameters:
        - frame_index: int, frame number in video
        - gaze, hand, trajectory: str, sensor interpretations
        - action: str, predicted target action
        """
        timestamp = round(datetime.now().timestamp(), 2)
        alert = "YES" if action in ["medication", "emergency_button"] else "NO"

        data = {
            "timestamp": timestamp,
            "frame": frame_index,
            "gaze": gaze,
            "hand": hand,
            "trajectory": trajectory,
            "predicted_action": action,
            "alert_flag": alert
        }

        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(data)
