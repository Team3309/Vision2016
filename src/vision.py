# coding=utf-8
# FRC Vision 2016
# Copyright 2016 Vinnie Magro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import cv2
import numpy as np

import vision_util as vision_common


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
    if contour_area > 0:
        diff = (bounding_area / contour_area) - (1.0 / 3.0)
        return 100 - (diff * 5)

    return 0


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


def get_corners(contour):
    """
    Given a contour that should have a rectangular convex hull, produce a sorted list of corners for the bounding rectangle
    :param contour:
    :return:
    """
    hull = cv2.convexHull(contour)
    hull_poly = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
    return sort_corners(hull_poly)


def sort_corners(corners):
    """
    Sorts a list of corners so that the top left corner is at index 0, top right at index 1 etc
    :param corners:
    :return:
    """
    if len(corners) != 4:
        raise ValueError('Incorrect number of corners')

    corners = [c[0] for c in corners]

    # sort corners by y value, pick the low 2 for top and high 2 for bottom
    corners = sorted(corners, key=lambda c: c[1])
    top = corners[:2]
    bottom = corners[2:]

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
    before_warp = np.zeros(bin_shape, np.uint8)
    cv2.drawContours(before_warp, [contour], -1, 255, -1)

    try:
        corners = get_corners(contour)

        # get a perspective transformation so that the target is warped as if it was viewed head on
        shape = (400, 280)
        dest_corners = np.array([(0, 0), (shape[0], 0), (0, shape[1]), (shape[0], shape[1])], np.float32)
        warp = cv2.getPerspectiveTransform(corners, dest_corners)
        fixed_perspective = cv2.warpPerspective(before_warp, warp, shape)
        fixed_perspective = fixed_perspective.astype(np.uint8)

        _, contours, _ = cv2.findContours(fixed_perspective, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_contour = contours[0]

        return new_contour, fixed_perspective

    except ValueError:
        raise ValueError('Failed to detect rectangle')


def contour_filter(contour, min_score, binary):
    """
    Threshold for removing hulls from positive detection.
    :return: True if it is a good match, False if it was a bad match (false positive).
    """
    # filter out particles less than 1000px^2
    bounding = cv2.boundingRect(contour)
    bounding_area = bounding[2] * bounding[3]
    contour_area = cv2.contourArea(contour)
    if bounding_area < 1000 or contour_area < 1000:
        return False

    try:
        contour, new_binary = fix_target_perspective(contour, binary.shape)
    except ValueError:
        return False

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
    corners = get_corners(contour)
    top_midpoint = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
    return top_midpoint


def target_distance(target):
    # http://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
    # distance to object (mm) = focal length (mm) * real height of the object (mm) * image height (pixels)
    #                           ---------------------------------------------------------------------------
    #                           object height (pixels) * sensor height (mm)
    target_height_mm = 304.8  # 1ft = 304.8mm
    focal_length_mm = 3.60  # raspberry pi camera module focal length
    sensor_height_mm = 2.74  # raspberry pi camera module sensor height in mm
    # image height is 100%, target[1][1] is the fraction height of the total image height
    image_height = 1
    target_img_height = target[1][1]

    distance_mm = (focal_length_mm * target_height_mm * image_height) / (target_img_height * sensor_height_mm)
    dist_inches = distance_mm * 0.039370
    return dist_inches


def target_angle_of_elevation(dist_in):
    delta_height = 96.5 - 13  # 96.5 inches off the ground - 13 inches height of shooter
    return math.degrees(math.atan2(delta_height, dist_in))


def target_azimuth(target):

    # target[1][0] is [0,1] percentage of full frame width
    # full frame width =    target_width_in
    #                       -------------------------------
    #                       target_width%
    # target is 1ft8in wide
   # full_frame_width_in = (12 + 8) / target[1][0]
   # center_x_in = full_frame_width_in / 2
    # target x inches = (1/2)(target_x + 1)
    #                   -------------------
    #                   full_width_in
   # target_x_in = ((1 / 2) * (target[0][0] + 1)) / full_frame_width_in
   # delta_x_in = target_x_in - center_x_in
   # azimuth = math.atan2(delta_x_in, target_distance(target))
   # azimuth = math.degrees(azimuth)
   # return azimuth
    goal_width_percent = target[1][0]
    x = target[0][0]
    percent_to_turn = (x/2)

    width_of_image_in = (8 + 12)/goal_width_percent
    delta_x_inches = width_of_image_in * percent_to_turn
    return math.degrees( math.atan2(delta_x_inches,target_distance(target)))



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
    Detect high goals in the input image
    :param img: hsv input image
    :param hue_min:
    :param hue_max:
    :param sat_min:
    :param sat_max:
    :param val_min:
    :param val_max:
    :param output_images: images that show the output of various stages of the detection process
    :return: a list of the detected targets
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
    print 'contour filtered', original_count, 'to', len(filtered_contours)

    # convert img back to bgr so it looks good when displayed
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # draw outlines so we know it actually detected it
    polys = [cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) for contour in filtered_contours]
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
            'distance': target_distance(target),
            'elevation_angle': target_angle_of_elevation(target_distance(target)),
            'azimuth': target_azimuth(target)
        } for target in targets]

    return output_targets
