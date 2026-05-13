def map_point_to_target(x, y, x1, y1, x2, y2):
    if x2 <= x1 or y2 <= y1:
        return None

    nx = (x - x1) / float(x2 - x1)
    ny = (y - y1) / float(y2 - y1)

    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))

    return nx, ny


def point_inside_target(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2