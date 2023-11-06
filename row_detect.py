import cv2 as cv
import random
from row_helpers import binary_simple, find_best_counts, draw_peak_lines

# Option 1: Use specific image file in ./data directory
filename = "crop_row_258.JPG"

# Option 2: Use random image from CRBD dataset if filename not specified
if filename == "":
    filename = f"crop_row_{random.randint(1, 281):03}.JPG"
    filepath = f"./data/CRBD/Images/{filename}"
else:
    filepath = f"./data/{filename}"

# Read image file into OpenCV/numpy formal
img = cv.imread(filepath, flags=cv.IMREAD_COLOR)

# Convert to HSV and calculate binary thresholded image (find green pixels)
binary_img = binary_simple(img)

# Calculate height and width of image
(height, width) = binary_img.shape[:2]

# Find best row estimate for each bottom pixel
top, count = find_best_counts(binary_img)

# Find best estimates for rows and add to image
draw_peak_lines(img, count, top, height)

# Display image with row estimates
cv.imshow("Display window", img)

# Display binary image
cv.imshow("hsv_binary window", binary_img)

# Export images
cv.imwrite(f"./data/outputs/{filename}", img)
cv.imwrite(f"./data/outputs/binary_{filename}", binary_img)

k = cv.waitKey(0)
