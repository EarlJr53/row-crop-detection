import cv2 as cv
from skimage.draw import line as skidline
from scipy.signal import argrelextrema, savgol_filter
import numpy as np
import matplotlib.pyplot as plt

def binary_adaptive(img):
    # adaptive_img = cv.adaptiveThreshold(hsv_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img

def binary_simple(img):
    h_low = 40
    h_upper = 80
    s_low = 50
    s_upper = 215
    v_low = 60
    v_upper = 200

    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_binary_image = cv.inRange(
        hsv_image,
        (h_low, s_low, v_low),
        (h_upper, s_upper, v_upper)
    )

    return hsv_binary_image

def find_best_counts(binary_img):
    (height, width) = binary_img.shape[:2]
    best_count = np.zeros(width)
    best_top = np.zeros(width)
    for bottom in range(width):
        for top in range(width):
            rr, cc = skidline(height - 1, bottom, 0, top)
            count = cv.countNonZero(binary_img[rr,cc])
            if count > best_count[bottom]:
                best_top[bottom] = top
                best_count[bottom] = count
    return best_top, best_count

def draw_peak_lines(img, count, top, height):
    peaks = argrelextrema(savgol_filter(count, 10, 5), np.greater, order=30)
    print(peaks)

    for bottom_peak in peaks[0]:
        cv.line(img, (int(bottom_peak), height - 1), (int(top[int(bottom_peak)]), 0), (0,0,255), 4)
        # cv.line(mask, (height - 1, int(bottom_peak)), (0, int(best_top[int(bottom_peak)])), (0,0,255), 4)