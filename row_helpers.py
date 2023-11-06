import cv2 as cv
from skimage.draw import line as skidline
from scipy.signal import argrelextrema, savgol_filter
from skspatial.objects import Line
import numpy as np
import matplotlib.pyplot as plt


def binary_simple(img):
    """
    Takes BGR image, converts to HSV, and then using preset threshold values,
    converts to binary image, selecting green sections of image.

    Args:
        img (MatLike): BGR image of a field

    Returns:
        MatLike: binary image (green sections of field)
    """

    # Hue thresholds
    h_low = 35
    h_upper = 85

    # Saturation thresholds
    s_low = 50
    s_upper = 215

    # Value thresholds
    v_low = 60
    v_upper = 200

    # Converts BGR inage to HSV
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Converts HSV image to binary based on hard-coded thresholds
    hsv_binary_image = cv.inRange(
        hsv_image, (h_low, s_low, v_low), (h_upper, s_upper, v_upper)
    )

    return hsv_binary_image


def find_best_counts(binary_img):
    """
    Determines best line for each pixel in the bottom row

    Args:
        binary_img (MatLike): Binary image with green regions selected

    Returns:
        NDArray: array of top row pixels that correspond to the best line
            originating from each bottom pixel
        NDArray: array of number of green pixels for the line originating at
            each bottom pixel
    """

    # Get shape of image
    (height, width) = binary_img.shape[:2]

    # Initialize arrays to hold counts and top points
    best_count = np.zeros(width)
    best_top = np.zeros(width)

    # Loop through bottom and top points
    for bottom in range(width):
        for top in range(width):
            # Calculate line between 2 given pixels
            rr, cc = skidline(height - 1, bottom, 0, top)

            # Count nonzero (green) points along line
            # divide by total number of points along line
            count = cv.countNonZero(binary_img[rr, cc]) / len(rr)

            # Set new best line for this bottom point
            if count > best_count[bottom]:
                best_top[bottom] = top
                best_count[bottom] = count

    return best_top, best_count


def draw_peak_lines(img, count, top, height):
    """_summary_

    Args:
        img (MatLike): BGR image of field
        count (NPArray): array of green pixel ratio for all best lines
        top (NPArray): array of top ends of all best lines
        height (int): height of image
    """

    # Smooth peaks and choose relative maxima for green pixel ratio
    peaks = argrelextrema(
        savgol_filter(count, 20, 5), np.greater, order=int(len(count) / 5)
    )
    print(peaks)

    # For each row estimate, draw a line on the image
    for bottom_peak in peaks[0]:
        cv.line(
            img,
            (int(bottom_peak), height - 1),
            (int(top[int(bottom_peak)]), 0),
            (0, 0, 255),
            4,
        )

    # ! Not fully implemented: calculate route of robot toward vanishing point
    # bottom, target = calc_path(peaks, top)
    # all_lines = []
    # intersections = np.zeros
    # for bottom_peak in peaks[0]:
    #     all_lines.append(Line.from_points([bottom_peak, height - 1], [int(top[int(bottom_peak)]), 0]))
    # for line1 in all_lines:
    #     for line2 in all_lines:
    #         if line2 != line1:
    #             intersections.append(line1.intersect_line(line2))


def calc_path(peaks, top):
    """
    Calculates preferred path of robot (path to vanishing point)
    ! NOT FULLY IMPLEMENTED

    Args:
        peaks (_type_): _description_
        top (NPArray): array of top ends of all best lines

    Returns:
        _type_: _description_
    """
    # bottom = len(top)/2
    bottom = int(sum(peaks[0]) / len(peaks[0]))
    target = int(sum(top[peaks[0]]) / len(top[peaks[0]]))

    print(bottom, target)
    return bottom, target
