def get_hand_preshape(landmarks, w, h):
    """
    Estimate the hand preshape (grip) from hand landmarks.

    Parameters:
    - landmarks: Mediapipe hand landmarks
    - w, h: width and height of frame

    Returns:
    - str: 'gripping' if hand is preparing to grasp, 'open' otherwise
    """
    # Use thumb and index tip positions
    thumb = landmarks[4]
    index = landmarks[8]

    x1, y1 = int(thumb.x * w), int(thumb.y * h)
    x2, y2 = int(index.x * w), int(index.y * h)

    # Euclidean distance between thumb and index fingertips
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Adjusted threshold for elderly (allowing for reduced finger extension)
    return "gripping" if distance < 50 else "open"

# Future improvement:
# Add more hand shape types like 'support grip', 'resting', or 'tremor signature'
