"""
Basic usage example for the Elderly Monitoring System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main_predictor import main

if __name__ == "__main__":
    print("Starting Elderly Monitoring System...")
    print("Press 'q' to quit")
    main()
