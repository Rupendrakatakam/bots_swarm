# geometry.py
import math
import numpy as np

def get_closest_point_on_circle(q, circle):
    """Finds the closest point on the surface of a circular obstacle."""
    cx, cy, radius = circle
    center = np.array([cx, cy], dtype=float)
    dist = np.linalg.norm(q - center)
    
    if dist == 0: return center
        
    direction = (q - center) / dist
    return center + direction * radius

def get_closest_point_on_rect(q, rect):
    """Finds the closest point on the surface of a rotated rectangular obstacle."""
    cx, cy, L, W, theta_deg = rect
    theta = math.radians(theta_deg)
    
    # 1. Vector from rectangle center to robot
    dx = q[0] - cx
    dy = q[1] - cy
    
    # 2. Rotate vector to match the rectangle's local coordinate frame
    lx = dx * math.cos(theta) + dy * math.sin(theta)
    ly = -dx * math.sin(theta) + dy * math.cos(theta)
    
    # 3. Clamp the point to the physical boundaries of the rectangle
    px = max(-L/2.0, min(L/2.0, lx))
    py = max(-W/2.0, min(W/2.0, ly))
    
    # 4. Rotate the closest point back to global coordinates
    qx = px * math.cos(-theta) - py * math.sin(-theta) + cx
    qy = px * math.sin(-theta) + py * math.cos(-theta) + cy
    
    return np.array([qx, qy])