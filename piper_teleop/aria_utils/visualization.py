"""OpenCV overlays for calibrated Aria gaze and hand projections."""


HAND_CONNECTIONS = (
    (5, 17), (17, 18), (18, 19), (19, 4),
    (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 11), (11, 12), (12, 13), (13, 2),
    (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 6), (6, 7), (7, 0),
    (6, 8), (8, 11), (11, 14), (14, 17),
)


def draw_perception_bgr(image, projection):
    """Draw one gaze target and two 21-landmark hand skeletons in-place."""

    import cv2

    height, width = image.shape[:2]

    def visible(point):
        return point is not None and 0 <= point[0] < width and 0 <= point[1] < height

    for side, color in (("left", (80, 255, 80)), ("right", (255, 80, 255))):
        points = projection.get(side, [None] * 21)
        for first, second in HAND_CONNECTIONS:
            if visible(points[first]) and visible(points[second]):
                cv2.line(
                    image,
                    tuple(round(value) for value in points[first]),
                    tuple(round(value) for value in points[second]),
                    color,
                    3,
                    cv2.LINE_AA,
                )
        for point in points:
            if visible(point):
                cv2.circle(
                    image,
                    tuple(round(value) for value in point),
                    4,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
    gaze = projection.get("gaze")
    if visible(gaze):
        center = tuple(round(value) for value in gaze)
        cv2.circle(image, center, 18, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(image, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(
            image,
            (center[0] - 25, center[1]),
            (center[0] + 25, center[1]),
            (0, 255, 255),
            2,
        )
        cv2.line(
            image,
            (center[0], center[1] - 25),
            (center[0], center[1] + 25),
            (0, 255, 255),
            2,
        )
    return image
