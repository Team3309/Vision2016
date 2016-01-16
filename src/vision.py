# coding=utf-8
import math

import cv2
import numpy as np

import vision_util as vision_common

# limit blobs to at least 1000px^2
min_size = 1000


def aspect_ratio_score(contour):
    rect = cv2.minAreaRect(contour)
    width = rect[1][0]
    height = rect[1][1]

    ratio_score = 0.0

    # check to make sure the size is defined to prevent possible division by 0 error
    if width != 0 and height != 0:
        # the target is 1ft8in wide by 1ft2in high, so ratio of width/height is 20/14
        ratio_score = 100 - abs((width / height) - (20 / 14))

    return ratio_score


def coverage_score(contour):
    contour_area = cv2.contourArea(contour)
    bounding_size = cv2.boundingRect(contour)
    bounding_area = bounding_size[2] * bounding_size[3]
    # ideal area is 1/3
    denominator = contour_area - (1 / 3)
    if denominator == 0:
        return 100
    else:
        return 100 - (bounding_area / denominator)


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
    Calculate a score based on the "profile" of the target, basically how closely its geometry matches with the expected
    geometry of the goal
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


def sort_corners(corners):
    """
    Sorts a list of corners so that the top left corner is at index 0, top right at index 1 etc
    :param corners:
    :return:
    """
    center_y = 0
    for corner in corners:
        center_y += corner[0][1]

    center_y /= len(corners)

    top = []
    bottom = []
    center = []
    for corner in corners:
        if corner[0][1] < center_y:
            top.append(corner[0])
        elif corner[0][1] > center_y:
            bottom.append(corner[0])
        else:
            center.append(corner[0])
    for point in center:
        if len(top) == 1:
            top.append(point)
        elif len(bottom) == 1:
            bottom.append(point)

    tl = top[1] if top[0][0] > top[1][0] else top[0]
    tr = top[0] if top[0][0] > top[1][0] else top[1]
    bl = bottom[1] if bottom[0][0] > bottom[1][0] else bottom[0]
    br = bottom[0] if bottom[0][0] > bottom[1][0] else bottom[1]
    return np.array([tl, tr, bl, br], np.float32)


def fix_target_perspective(contour, bin_shape):
    """
    Fixes the perspective so it always looks as if we are viewing it head-on
    :param contour:
    :param bin_shape: numpy shape of the binary image matrix
    :return: a new version of contour with corrected perspective, a new binary image to test against,
    """
    hull = cv2.convexHull(contour)
    poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
    corners = sort_corners(poly)
    before_warp = np.zeros(bin_shape)
    cv2.drawContours(before_warp, [contour], -1, 255, -1)

    # get a perspective transformation so that the target is warped as if it was viewed head on
    shape = (400, 280)
    warp = cv2.getPerspectiveTransform(corners, np.array([(0, 0), (shape[0], 0), (0, shape[1]), (shape[0], shape[1])],
                                                         np.float32))
    fixed_perspective = cv2.warpPerspective(before_warp, warp, shape)
    fixed_perspective = fixed_perspective.astype(np.uint8)

    _, contours, _ = cv2.findContours(fixed_perspective, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_contour = contours[0]

    return new_contour, fixed_perspective


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

    contour, new_binary = fix_target_perspective(contour, binary.shape)

    scores = np.array([
        aspect_ratio_score(contour),
        coverage_score(contour),
        moment_score(contour),
        profile_score(contour, new_binary)
    ])

    if scores.mean() < min_score:
        return False

    return True


def target_center(contour):
    """
    Calculate center point, the midpoint between the upper two corners
    :param contour:
    :return:
    """
    hull = cv2.convexHull(contour)
    poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
    corners = sort_corners(poly)
    top_midpoint = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
    return top_midpoint


def target_distance(target):
    target_inches = 20  # 1ft8in
    fov = math.radians(37.4)
    # d = Tft*FOVpixel/(2*Tpixel*tanΘ)
    # can ignore pixels here because width is normalized to be [0,1] in terms of percentage of the image
    dist_inches = target_inches * 1 / (2 * target[1][0] * math.tan(fov))
    return dist_inches


def to_targeting_coords(target, imshape):
    """
    Convert to a targeting coordinate system of [-1, 1]
    :param target:
    :return:
    """
    imheight, imwidth, _ = imshape
    imheight = float(imheight)
    imwidth = float(imwidth)

    (x, y), (width, height) = target
    x -= imwidth / 2
    x /= imwidth / 2
    y -= imheight / 2
    y /= imheight / 2
    y *= -1
    width /= imwidth
    height /= imheight
    return (x, y), (width, height)


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

    output_images['bin'] = np.copy(bin)

    _, contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter out so only left with good contours
    original_count = len(contours)
    filtered_contours = [x for x in contours if contour_filter(contour=x, min_score=95, binary=bin)]
    print 'contour filtered ', original_count, ' to ', len(filtered_contours)
    polys = map(lambda contour: cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True), filtered_contours)

    # convert img back to bgr so it looks good when displayed
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # draw outlines so we know it actually detected it
    cv2.drawContours(img, polys, -1, (0, 0, 255), 2)

    original_targets = [(target_center(contour), cv2.boundingRect(contour)) for contour in filtered_contours]
    original_targets = [(center, (rect[2], rect[3])) for (center, rect) in original_targets]
    # original_targets is now a list of (x, y) and (width, height)
    targets = [to_targeting_coords(target, img.shape) for target in original_targets]

    # draw targeting coordinate system on top of the result image
    # axes
    imheight, imwidth, _ = img.shape
    cv2.line(img, (int(imwidth / 2), 0), (int(imwidth / 2), int(imheight)), (255, 255, 255), 5)
    cv2.line(img, (0, int(imheight / 2)), (int(imwidth), int(imheight / 2)), (255, 255, 255), 5)
    # aiming reticle
    cv2.circle(img, (int(imwidth / 2), int(imheight / 2)), 50, (255, 255, 255), 5)

    # draw dots on the center of each target
    for target in original_targets:  # use original_targets so we don't have to recalculate image coords
        x = int(target[0][0])
        y = int(target[0][1])
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

    output_images['result'] = img

    output_targets = [
        {
            'pos': {
                'x': target[0][0],
                'y': target[0][1]
            },
            'size': {
                'width': target[1][0],
                'height': target[1][1]
            },
            'distance': target_distance(target)
        } for target in targets]

    return output_targets
