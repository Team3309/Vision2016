import cv2
import numpy as np
import math


def hsv_threshold(img, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
    """
    Threshold an HSV image given separate min/max values for each channel.
    :param img: an hsv image
    :param hue_min:
    :param hue_max:
    :param sat_min:
    :param sat_max:
    :param val_min:
    :param val_max:
    :return: result of the threshold (each binary channel AND'ed together)
    """

    hue, sat, val = cv2.split(img)

    hue_bin = np.zeros(hue.shape, dtype=np.uint8)
    sat_bin = np.zeros(sat.shape, dtype=np.uint8)
    val_bin = np.zeros(val.shape, dtype=np.uint8)

    cv2.inRange(hue, hue_min, hue_max, hue_bin)
    cv2.inRange(sat, sat_min, sat_max, sat_bin)
    cv2.inRange(val, val_min, val_max, val_bin)

    bin = np.copy(hue_bin)
    cv2.bitwise_and(sat_bin, bin, bin)
    cv2.bitwise_and(val_bin, bin, bin)

    return bin


def canny(img, lowThreshold):
    """
    Performs canny edge detection on the provided grayscale image.
    :param img: a grayscale image
    :param lowThreshold: threshold for the canny operation
    :return: binary image containing the edges found by canny
    """

    dst = np.zeros(img.shape, dtype=img.dtype)
    cv2.blur(img, (3, 3), dst)

    # canny recommends that the high threshold be 3 times the low threshold
    # the kernel size is 3 as defined above
    return cv2.Canny(dst, lowThreshold, lowThreshold * 3, dst, 3)


def convex_hulls(contours):
    """
    Convenience method to get a list of convex hulls from list of contours
    :param contours: contours that should be turned into convex hulls
    :return: a list of convex hulls that match each contour
    """

    hulls = []
    for contour in contours:
        hulls.append(cv2.convexHull(contour))

    return hulls


def angle(rect):
    """
    Produce a more useful angle from a rotated rect. This format only exists to make more sense to the programmer or
    a user watching the output. The algorithm for transforming the angle is to add 180 if the width < height or
    otherwise add 90 to the raw OpenCV angle.
    :param rect: rectangle to get angle from
    :return: the formatted angle
    """
    if rect[1][0] < rect[1][1]:
        return rect[2] + 180
    else:
        return rect[2] + 90


def distance(a, b):
    """
    Calculate the distance between points a & b
    :return: distance as given by the distance formula: sqrt[(a.x - b.x)^2 + (a.y - b.y)^2]
    """
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - a[2], 2))
