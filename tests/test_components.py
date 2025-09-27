"""
Basic tests for the Elderly Monitoring System components
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaze import get_gaze_direction
from hand import get_hand_preshape
from trajectory import get_trajectory_direction

class TestElderlyMonitoring(unittest.TestCase):
    
    def test_gaze_direction(self):
        """Test gaze direction detection"""
        # Mock landmarks for testing
        class MockLandmark:
            def __init__(self, x, y, z=0):
                self.x = x
                self.y = y
                self.z = z
        
        # Create mock landmarks
        landmarks = [MockLandmark(0, 0) for _ in range(500)]
        landmarks[33].x = 0.2   # Left eye corner
        landmarks[133].x = 0.8  # Right eye corner
        landmarks[468].x = 0.5  # Iris center
        
        result = get_gaze_direction(landmarks, 640, 480)
        self.assertIn(result, ['left', 'center', 'right'])
    
    def test_hand_preshape(self):
        """Test hand gesture recognition"""
        class MockLandmark:
            def __init__(self, x, y, z=0):
                self.x = x
                self.y = y
                self.z = z
        
        # Create mock hand landmarks
        landmarks = [MockLandmark(0, 0) for _ in range(21)]
        landmarks[4].x = 0.5   # Thumb tip
        landmarks[4].y = 0.5
        landmarks[8].x = 0.6   # Index tip (close to thumb = gripping)
        landmarks[8].y = 0.5
        
        result = get_hand_preshape(landmarks, 640, 480)
        self.assertIn(result, ['gripping', 'open'])
    
    def test_trajectory_direction(self):
        """Test trajectory direction detection"""
        # Test with insufficient data
        result = get_trajectory_direction([(100, 100)])
        self.assertEqual(result, 'idle')
        
        # Test with movement data
        trajectory = [(100, 100), (120, 100), (140, 100), (160, 100), (180, 100)]
        result = get_trajectory_direction(trajectory)
        self.assertIn(result, ['left', 'right', 'center'])

if __name__ == '__main__':
    unittest.main()
