def get_trajectory_direction(trajectory):
    """
    Determine horizontal direction of arm movement from trajectory points.

    Parameters:
    - trajectory: list of (x, y) tuples representing past wrist positions

    Returns:
    - str: one of ['left', 'center', 'right', 'idle']
    """
    if len(trajectory) < 5:
        return "idle"  # Not enough data to estimate movement

    x_start = trajectory[0][0]
    x_end = trajectory[-1][0]
    x_diff = x_end - x_start

    # Reduced threshold for subtle elderly motion
    if x_diff > 20:
        return "right"
    elif x_diff < -20:
        return "left"
    else:
        return "center"

# Future extension:
# Add vertical movement tracking (e.g., reaching down or up) or detect sudden jerks/falls.
