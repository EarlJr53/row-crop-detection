import cv2 as cv
import random
from row_helpers import binary_simple, binary_adaptive, find_best_counts, draw_peak_lines

image_number = random.randint(1, 281)
img = cv.imread(f"./data/CRBD/Images/crop_row_{image_number:03}.JPG", flags=cv.IMREAD_COLOR)

binary_img = binary_simple(img)

(height, width) = binary_img.shape[:2]

# mask = np.zeros((height, width))

top, count = find_best_counts(binary_img)

draw_peak_lines(img, count, top, height)

cv.imshow("Display window", img)
cv.imshow("hsv_binary window", binary_img)
# cv.imshow("mask", mask)

k = cv.waitKey(0)
