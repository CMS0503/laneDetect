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

# cv2.imshow("Original", img)

cap = cv2.VideoCapture('../challenge_video.mp4')
        # while(cap.isOpened()):

while True:
    _, frame = cap.read()

    low = cv2.getTrackbarPos('low threshold', 'bin')
    high = cv2.getTrackbarPos('high threshold', 'bin')
    s_thresh = (low, high)
    img_binary = find_edges(frame, s_thresh)

    output = np.zeros((720, 1400, 3), dtype=np.uint8)

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img_binary = cv2.resize(img_binary, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    img_binary = np.reshape(img_binary, (360, 640, 1))
    print(img_binary.shape)
    w = frame.shape[1]
    h = frame.shape[0]

    output[40:40+h, 20:20+w, :] = frame
    output[40:40+h, 20*2+w:20*2+w*2, :] = img_binary

    cv2.imshow("bin", output)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()