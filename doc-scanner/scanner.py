import os

import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local

from exceptions import ImageIsNoneException, FailedFindEdgesException

current_dir = os.getcwd()
assets_path = os.path.join(current_dir, "images", )


def run_scan_by_path(path):
    _img = _read_and_transform_image(path)
    return _process_image(_img)


def run_scan_by_image(image):
    return _process_image(image)


def _order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def _calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _four_points_transform(image, points):
    rect = _order_points(points)
    (top_left, top_right, bot_right, bot_left) = rect

    width_top = _calculate_distance(top_left, top_right)
    width_bottom = _calculate_distance(bot_left, bot_right)
    max_width = max(int(width_top), int(width_bottom))

    height_left = _calculate_distance(top_left, bot_left)
    height_right = _calculate_distance(top_right, bot_right)
    max_height = max(int(height_left), int(height_right))

    destination_coords = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]]
        , dtype="float32")

    transformation_matrix = cv2.getPerspectiveTransform(rect, destination_coords)
    t_image = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))

    return t_image


def _read_and_transform_image(image_name):
    path = f"{assets_path}{os.sep}{image_name}"
    print(path)
    image = cv2.imread(path)
    return image


def _detect_edges(image, threshold_lower=50, threshold_upper=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged_image = cv2.Canny(gray, threshold_lower, threshold_upper)
    return edged_image


def _find_contours(edged_image):
    contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    width, height = edged_image.shape[1], edged_image.shape[0]
    image_contour = np.array([[(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]])
    screen_contour = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        if len(approx) == 4:
            screen_contour = approx
            break

    if screen_contour is None:
        screen_contour = image_contour
    else:
        area = cv2.contourArea(screen_contour)
        img_area = cv2.contourArea(image_contour)
        if img_area * 0.15 > area:
            return image_contour
    return screen_contour


def _apply_transformation(original_image, screen_contour, ratio):
    warped = _four_points_transform(original_image, screen_contour.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    threshold = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > threshold).astype("uint8") * 255
    return warped


def _process_image(image):
    if image is None:
        raise ImageIsNoneException
    ratio = image.shape[0] / 500.0
    original_image = image.copy()
    image = imutils.resize(image, height=500)

    edged_image = _detect_edges(image, 75, 200)
    screen_contour = _find_contours(edged_image)

    if screen_contour is None:
        raise FailedFindEdgesException

    t_img = _apply_transformation(original_image, screen_contour, ratio)
    return t_img
