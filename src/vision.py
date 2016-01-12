import math

import cv2
import numpy as np

import vision_util as vision_common

# limit blobs to at least 1000px^2
min_size = 1000


def aspect_ratio_score(contour):
    """
    Give a score to a convex hull based on how likely it is to be a top goal.
    :param hull: convex hull to test
    :return: Score based on the ratio of side lengths and a minimum area
    """
    rect = cv2.minAreaRect(contour)
    width = rect[1][0]
    height = rect[1][1]

    ratio_score = 0.0

    # check to make sure the size is defined to prevent possible division by 0 error
    if width != 0 and height != 0:
        # the target is 1ft8in wide by 1ft2in high, so ratio of width/height is 1.429
        ratio_score = 100 - abs((width / height) - 1.429)

    return ratio_score


def side_score(contour):
    """
    Give a score for a contour based on how likely it is to be a high goal
    :param contour:
    :return:
    """
    if cv2.contourArea(contour) < min_size:
        return 0
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # the goal marker has 8 sides
    side_count = len(approx)
    if side_count == 8:
        return 100
    return 0


def coverage_score(contour):
    contour_area = cv2.contourArea(contour)
    bounding_size = cv2.boundingRect(contour)
    bounding_area = bounding_size[2] * bounding_size[3]
    # ideal area is 1/3
    return 100 - (bounding_area / contour_area - (1 / 3))


def moment_score(contour):
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments)
    # hu[6] should be close to 0
    return 100 - (hu[6] * 100)


def col_profile(num_cols, height):
    profile = np.zeros(num_cols)
    peak_width = int(math.ceil(num_cols * 0.125))

    # average number of pixels should be height
    for i in range(0, peak_width):
        profile[i] = height
    # average number of pixels should be 10% of height
    for i in range(peak_width, num_cols - peak_width):
        profile[i] = height * .1
    # average number of pixels should be height
    for i in range(num_cols - peak_width, num_cols):
        profile[i] = height

    # normalize to between 0 and 1
    profile *= 1.0 / profile.max()
    return profile


def row_profile(num_rows, width):
    profile = np.zeros(num_rows)
    # this is roughly equal to the width of the tape, which is where we will see a peak at the bottom of the image
    peak_width = int(math.ceil(num_rows * 0.125))

    # average number of pixels for first part should be 2*peak width, which is roughly 2*width of the tape
    for i in range(0, num_rows - peak_width):
        profile[i] = 2 * peak_width
    # peak at the end that takes up the whole image width
    for i in range(num_rows - peak_width, num_rows):
        profile[i] = width

    # normalize to between 0 and 1
    profile *= 1.0 / profile.max()
    return profile


def profile_score(contour, binary):
    """
    Calculate a score based on the "profile" of the target, basically how closely its geometry matches with the expected geometry of the goal
    :param contour:
    :param binary:
    :return:
    """
    bounding = cv2.boundingRect(contour)
    pixels = np.zeros((binary.shape[0], binary.shape[1]))
    cv2.drawContours(pixels, [contour], -1, 255, -1)
    col_averages = np.mean(pixels, axis=0)[bounding[0]:bounding[0] + bounding[2]]
    row_averages = np.mean(pixels, axis=1)[bounding[1]:bounding[1] + bounding[3]]
    # normalize to between 0 and 1
    col_averages *= 1.0 / col_averages.max()
    row_averages *= 1.0 / row_averages.max()

    col_diff = np.subtract(col_averages, col_profile(col_averages.shape[0], bounding[2]))
    row_diff = np.subtract(row_averages, row_profile(row_averages.shape[0], bounding[3]))

    # average difference should be close to 0
    avg_diff = np.mean([np.mean(col_diff), np.mean(row_diff)])
    return 100 - (avg_diff * 50)


def contour_filter(contour, min_score, binary):
    """
    Threshold for removing hulls from positive detection.
    :return: True if it is a good match, False if it was a bad match (false positive).
    """
    # filter out particles less than 1000px^2
    bounding = cv2.boundingRect(contour)
    bounding_area = bounding[2] * bounding[3]
    if bounding_area < 1000:
        return False

    aspect = aspect_ratio_score(contour)
    if aspect < min_score or math.isnan(aspect):
        return False
    coverage = coverage_score(contour)
    if coverage < min_score or math.isnan(coverage):
        return False
    moment = moment_score(contour)
    if moment < min_score:
        return False
    profile = profile_score(contour, binary)
    if profile < min_score:
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
    filtered_contours = [x for x in contours if contour_filter(contour=x, min_score=95, binary=bin)]
    print 'contour filtered ', original_count, ' to ', len(filtered_contours)
    polys = map(lambda contour: cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True), filtered_contours)

    # draw pink lines on all contours
    cv2.drawContours(img, contours, -1, (203, 192, 255), -1)
    # draw green outlines so we know it actually detected it
    cv2.drawContours(img, polys, -1, (255, 0, 0), 2)

    # draw scores on the hulls
    # for hull in hulls:
    #     center = cv2.minAreaRect(hull)[0]
    #     score = str(hull_score(hull))
    #     cv2.putText(img, score, center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 1)

    original_rects = map(lambda contour: cv2.boundingRect(contour), filtered_contours)
    # convert to the targeting system of [-1, 1]
    imheight, imwidth, _ = img.shape
    imheight = float(imheight)
    imwidth = float(imwidth)
    rects = map(lambda rect: (float(rect[0] / imwidth), float(rect[1] / imheight), rect[2], rect[3]), original_rects)
    rects = map(lambda rect: ((rect[0] * 2) - 1, -(rect[1] * 2) + 1, rect[2], rect[3]), rects)

    # draw targeting coordinate system on top of the result image
    # axes
    cv2.line(img, (int(imwidth / 2), 0), (int(imwidth / 2), int(imheight)), (255, 255, 255), 5)
    cv2.line(img, (0, int(imheight / 2)), (int(imwidth), int(imheight / 2)), (255, 255, 255), 5)
    # aiming reticle
    cv2.circle(img, (int(imwidth / 2), int(imheight / 2)), 100, (255, 255, 255), 5)

    # draw dots on the center of each target
    for rect in original_rects:  # use original_rects so we don't have to recalculate image coords
        x = rect[0] + (rect[2] / 2)
        y = rect[1]
        cv2.circle(img, (x, y), 20, (0, 0, 255), -1)

    output_images['result'] = img

    targets = map(lambda rect: {'pos': {'x': rect[0], 'y': rect[1]}, 'size': {'width': rect[2], 'height': rect[3]}},
                  rects)

    return targets


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
