import cv2 as cv
from skimage.draw import line as skidline
from scipy.signal import argrelextrema, savgol_filter
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./data/CRBD/Images/crop_row_200.JPG", flags=cv.IMREAD_COLOR)

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

# adaptive_img = cv.adaptiveThreshold(hsv_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

(height, width) = hsv_binary_image.shape[:2]
print(height, width)

mask = np.zeros((height, width))
best_count = np.zeros(width)
best_top = np.zeros(width)
# best_rr = np.zeros((width, height))
# best_cc = np.zeros((width, height))

for bottom in range(width):
    for top in range(width):
        rr, cc = skidline(height - 1, bottom, 0, top)
        count = cv.countNonZero(hsv_binary_image[rr,cc])
        if count > best_count[bottom]:
            best_top[bottom] = top
            best_count[bottom] = count
    
    # if best_count[bottom] > 60:
    # mask[best_rr, best_cc] = best_count[bottom]
    img[best_rr, best_cc] = (0, 0, best_count[bottom]*2)
    # print(best_count[bottom])

peaks = argrelextrema(savgol_filter(best_count, 10, 3), np.greater, order=20)
print(peaks)

for bottom_peak in peaks[0]:
    cv.line(img, (int(bottom_peak), height - 1), (int(best_top[int(bottom_peak)]), 0), (0,0,255), 4)
    # cv.line(mask, (height - 1, int(bottom_peak)), (0, int(best_top[int(bottom_peak)])), (0,0,255), 4)


# # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# # Set flags (Just to avoid line break in the code)
# flags = cv.KMEANS_RANDOM_CENTERS
# # Apply KMeans

# print(np.float32(best_count))
# compactness,labels,centers = cv.kmeans(np.float32(best_count),4,None,criteria,20,flags)

# print(centers)

# mask[0, centers] = 255

cv.imshow("Display window", img)
cv.imshow("hsv_binary window", hsv_binary_image)
cv.imshow("mask", mask)

# plt.plot(best_count)
# plt.show()

k = cv.waitKey(0)  # Wait for a keystroke in the window




# for bottompoint, toppoint in enumerate(top):
#     cv.line(img, (bottompoint, height - 1), (int(toppoint), 0), (0,0,count[bottompoint]*260), 1)