import cv2
from detect_lane import find_edges, get_binary
import numpy as np

def nothing():
    pass



img_gray = cv2.imread('../test_images/test1.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.imread('../test_images/test1.jpg')
# hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
# channel_s = hls[:, :, 2]

cv2.namedWindow("bin")
cv2.createTrackbar('low threshold', 'bin', 0, 255, nothing)
cv2.createTrackbar('high threshold', 'bin', 0, 255, nothing)

cv2.setTrackbarPos('low threshold', 'bin', 50)
cv2.setTrackbarPos('high threshold', 'bin', 150)

cv2.imshow("Original", img)

while True:

    low = cv2.getTrackbarPos('low threshold', 'bin')
    high = cv2.getTrackbarPos('high threshold', 'bin')
    s_thresh = (low, high)
    img_binary = find_edges(img, s_thresh)
    cv2.imshow("bin", img_binary)

    if cv2.waitKey(1)&0xFF == 27:
        break


cv2.destroyAllWindows()