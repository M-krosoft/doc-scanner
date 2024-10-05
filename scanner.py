import os

import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local

from exceptions import ImageIsNoneException, FailedFindEdgesException

current_dir = os.getcwd()
assets_path = os.path.join(current_dir, "images", )


def run_scan_by_path(path):
    img = read_and_transform_image(path)
    return process_image(img)


def run_scan_by_image(image):
    return process_image(image)


# order entry points: top left -> top right -> bot right -> bot left
def order_points(points):
    print(f'input={points}')
    rect = np.zeros((4, 2), dtype="float32")  # empty multidimensional array 4x2 of type float

    sums = points.sum(axis=1)  # calculate sum of x+y coords
    rect[0] = points[np.argmin(sums)]  # top left corner
    rect[2] = points[np.argmax(sums)]  # bot right corner

    diff = np.diff(points, axis=1)  # calculate diff of x-y coords
    rect[1] = points[np.argmin(diff)]  # top right corner
    rect[3] = points[np.argmax(diff)]  # bot left corner

    print(f'ordered_points={rect}')
    return rect


def add_image_borders(image):
    height, width = image.shape[:2]
    cv2.line(image, (0, 0), (width - 1, 0), (0, 0, 0), 2)
    cv2.line(image, (0, height - 1), (width - 1, height - 1), (0, 0, 0), 2)
    cv2.line(image, (0, 0), (0, height - 1), (0, 0, 0), 2)
    cv2.line(image, (width - 1, 0), (width - 1, height - 1), (0, 0, 0), 2)

    return image


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def four_points_transform(image, points):
    rect = order_points(points)
    (top_left, top_right, bot_right, bot_left) = rect

    width_top = calculate_distance(top_left, top_right)
    width_bottom = calculate_distance(bot_left, bot_right)
    max_width = max(int(width_top), int(width_bottom))  # width of transformed image

    height_left = calculate_distance(top_left, bot_left)
    height_right = calculate_distance(top_right, bot_right)
    max_height = max(int(height_left), int(height_right))  # height of transformed image

    destination_coords = np.array([
        [0, 0],  # new top left
        [max_width - 1, 0],  # new top right
        [max_width - 1, max_height - 1],  # new bot right
        [0, max_height - 1]]  # new bot left
        , dtype="float32")

    transformation_matrix = cv2.getPerspectiveTransform(rect, destination_coords)  # calculating transformation matrix for image
    t_image = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))  # transforming image

    return t_image


def read_and_transform_image(image_name):
    path = f"{assets_path}{os.sep}{image_name}"
    print(path)
    image = cv2.imread(path)
    return image


def detect_edges(image, threshold_lower=50, threshold_upper=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged_image = cv2.Canny(gray, threshold_lower, threshold_upper)  # return black and white image with edge as white
    return edged_image


def find_contours(edged_image):
    contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # search vertices of countour
    contours = imutils.grab_contours(contours)  # wyodrębnia kontury
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # choose 5 contours and sort descending

    width, height = edged_image.shape[1], edged_image.shape[0]
    image_contour = np.array([[(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]])
    screen_contour = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)  # calculates lenght of contour
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)  # approximate shape of polygon

        if len(approx) == 4:  # we got rectangle :)
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


def apply_transformation(original_image, screen_contour, ratio):
    warped = four_points_transform(original_image, screen_contour.reshape(4, 2) * ratio)  # transformation of countour shape
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # colors conversion
    threshold = threshold_local(warped, 11, offset=10, method="gaussian")  # local thresholding
    warped = (warped > threshold).astype("uint8") * 255  # init local thresholding
    return warped


def process_image(image):
    if image is None:
        raise ImageIsNoneException
    ratio = image.shape[0] / 500.0
    original_image = image.copy()
    image = imutils.resize(image, height=500)

    edged_image = detect_edges(image, 75, 200)
    screen_contour = find_contours(edged_image)

    # cv2.imshow("Edged", edged_image)
    #
    # cv2.drawContours(image, [screen_contour], -1, (0, 255, 0), 2)
    # cv2.imshow("CNT", image)
    # cv2.waitKey(0)

    if screen_contour is None:
        raise FailedFindEdgesException

    t_img = apply_transformation(original_image, screen_contour, ratio)
    return t_img


img = read_and_transform_image(image_name='1196-receipt.jpg')
transformed_image = process_image(img)
cv2.imshow("Contours", imutils.resize(img, height=650))
cv2.imshow("Scanned", imutils.resize(transformed_image, height=650))

cv2.waitKey(0)
cv2.destroyAllWindows()
