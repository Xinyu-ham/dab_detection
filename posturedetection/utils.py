import numpy as np


def get_angle_from_vertical(c1: int, c2: int) -> float:
    x1, y1 = c1
    x2, y2 = c2
    if y1 == y2:
        return 90 * np.sign(x1 - x2)
    
    dx = x1 - x2
    dy = y1 - y2
    rad = np.arctan(dx / dy)
    return rad / (2 * np.pi) * 360 + 180 * (0 if dy > 0 else 1)

def get_angle_at_joint(pt: tuple[int], left: tuple[int], right: tuple[int]) -> float:
    '''
        Calculate the angle in degrees in anti-clockwise direction of a joint but looking at the two adjacent joints connected.
        E.g. eblow bending degrees can be calculated from wrist to elbow and elbow to shoulder
    
        Params:
        ----------
        pt: tuple[int]
            joint where the angle is calculated
        left: tuple[int]
            The joint connected to pt that is away from the body center
        right = tuple[int]
            The joint connected to pt that is closer to the body center
    '''
    a, b, c = (np.array(x) for x in (pt, left, right))
    ab = b - a
    ba_norm = np.linalg.norm(ab)

    ac = c - a
    ac_norm = np.linalg.norm(ac)

    dot = np.dot(ab, ac)
    sign = np.sign(np.cross(ac, ab))
    return sign * np.arccos(dot / ba_norm/ ac_norm) / 2 / np.pi * 360

def angle_in_range(angle: float, target: float, margin: float) -> bool:
    high = target + margin
    low = target - margin
    
    if high >= 180:
        return angle >= low or angle <= high - 360
    elif low < -180:
        return angle <= high or angle >= low + 360
    else:
        return angle >= low and angle <= high
