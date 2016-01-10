import math

import cv2
import numpy as np

import vision_util as vision_common

# limit blobs to at least 1000px^2
min_size = 1000


def hull_score(hull):
    """
    Give a score to a convex hull based on how likely it is to be a top goal.
    :param hull: convex hull to test
    :return: Score based on the ratio of side lengths and a minimum area
    """
    rect = cv2.minAreaRect(hull)
    width = rect[1][0]
    height = rect[1][1]

    ratio_score = 0.0

    # check to make sure the size is defined to prevent possible division by 0 error
    if width != 0 and height != 0:
        # the target is 1ft8in wide by 1ft2in high, so ratio of width/height is 1.429
        ratio_score = 100 - abs((width / height) - 1.429)

    if cv2.contourArea(hull) < min_size:
        return 0

    return ratio_score


def hull_filter(hull, min_score=95):
    """
    Threshold for removing hulls from positive detection.
    :param hull: convex hull to test
    :return: True if it is a good match, False if it was a bad match (false positive).
    """
    score = hull_score(hull)
    if score < min_score or math.isnan(score):
        return False
    return True


def contour_score(contour):
    """
    Give a score for a contour based on how likely it is to be a high goal
    :param contour:
    :return:
    """
    if cv2.contourArea(contour) < min_size:
        return 0
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # the goal marker has 8 sides, we are allowing for 7 or 8 in case one is missing
    side_count = len(approx)
    if side_count == 8:
        return 100
    return 0


def contour_filter(contour, min_score=95):
    """
    Threshold for removing hulls from positive detection.
    :param hull: convex hull to test
    :return: True if it is a good match, False if it was a bad match (false positive).
    """
    score = contour_score(contour)
    if score < min_score or math.isnan(score):
        return False
    return True


def find(img, hue_min, hue_max, sat_min, sat_max, val_min, val_max, output_images):
    """
    Detect direction markers. These are the orange markers on the bottom of the pool that point ot the next objective.
    :param img: HSV image from the camera
    :return: a list of tuples indicating the rectangles detected (center, width x height, angle)
    """

    img = np.copy(img)

    bin = vision_common.hsv_threshold(img, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    # erode to remove bad dots
    erode_kernel = np.ones((1, 1), np.uint8)
    bin = cv2.erode(bin, erode_kernel, iterations=1)
    # dilate bin to fill any holes
    dilate_kernel = np.ones((5, 5), np.uint8)
    bin = cv2.dilate(bin, dilate_kernel, iterations=1)

    output_images['bin'] = bin

    canny = vision_common.canny(bin, 50)

    # find contours after first processing it with Canny edge detection
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter out so only left with good contours
    original_count = len(contours)
    filtered_contours = filter(contour_filter, contours)
    print 'contour filtered ', original_count, ' to ', len(filtered_contours)
    polys = map(lambda contour: cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True), filtered_contours)

    # now process based on convex hulls
    hulls = vision_common.convex_hulls(filtered_contours)
    cv2.drawContours(bin, hulls, -1, 255)
    original_count = len(hulls)

    # remove convex hulls that don't pass our scoring threshold
    hulls = filter(hull_filter, hulls)
    print 'hull filtered ', original_count, ' to ', len(hulls)

    # draw pink lines on all contours
    cv2.drawContours(img, contours, -1, (203, 192, 255), -1)
    # draw hulls in Blaze Orange
    cv2.drawContours(img, hulls, -1, (0, 102, 255), -1)
    # draw green outlines so we know it actually detected it
    cv2.drawContours(img, polys, -1, (255, 0, 0), 2)

    # draw scores on the hulls
    # for hull in hulls:
    #     center = cv2.minAreaRect(hull)[0]
    #     score = str(hull_score(hull))
    #     cv2.putText(img, score, center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 1)

    rects = map(lambda hull: cv2.minAreaRect(hull), hulls)
    # shape[0] is the number of rows because matrices are dumb
    rects = map(
        lambda rect: ((rect[0][1] / img.shape[1], rect[0][0] / img.shape[0]), rect[1], vision_common.angle(rect)),
        rects)
    # convert to the targeting system of [-1, 1]
    rects = map(lambda rect: (((rect[0][0] * 2) - 1, (rect[0][1] * 2) - 1), rect[1], rect[2]), rects)

    output_images['result'] = img

    return rects


def nothing(x):
    pass


if __name__ == '__main__':
    # cv2.namedWindow('bin')
    # cv2.createTrackbar('min', 'bin', 0, 255, nothing)
    # cv2.createTrackbar('max', 'bin', 0, 255, nothing)

    img = cv2.imread('/Users/vmagro/Desktop/tower.png', cv2.IMREAD_COLOR)
    # cv2.imshow('raw', img)

    # while True:
    # find(img, val_min=cv2.getTrackbarPos('min', 'bin'), val_max=cv2.getTrackbarPos('max', 'bin'))
    #     print(cv2.getTrackbarPos('min', 'bin'), cv2.getTrackbarPos('max', 'bin'))
    #     cv2.waitKey(1)

    output_images = {}
    rects = find(img, output_images=output_images)
    cv2.imshow('bin', output_images['bin'])

    cv2.imshow('result', output_images['result'])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
