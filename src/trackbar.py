import cv2
from detect_lane import find_edges, get_binary, dir_threshold, abs_sobel_thresh
import numpy as np

def nothing():
    pass

img = cv2.imread('../test_images/test12.jpg')
img = cv2.resize(img, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)

cv2.namedWindow("bin")

video = "origin"
type = 1

if type == 0:
    cap = cv2.VideoCapture('../challenge_video.mp4')

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

# white and yellow
elif type == 1:
    cap = cv2.VideoCapture(f'../{video}_video.avi')

    cv2.createTrackbar('low h', 'bin', 0, 180, nothing)
    cv2.createTrackbar('high h', 'bin', 0, 180, nothing)
    cv2.createTrackbar('low s', 'bin', 0, 255, nothing)
    cv2.createTrackbar('high s', 'bin', 0, 255, nothing)
    cv2.createTrackbar('low v', 'bin', 0, 255, nothing)
    cv2.createTrackbar('high v', 'bin', 0, 255, nothing)

    cv2.setTrackbarPos('low h', 'bin', 0)
    cv2.setTrackbarPos('high h', 'bin', 255)
    cv2.setTrackbarPos('low s', 'bin', 0)
    cv2.setTrackbarPos('high s', 'bin', 255)
    cv2.setTrackbarPos('low v', 'bin', 200)
    cv2.setTrackbarPos('high v', 'bin', 255)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



        low_h = cv2.getTrackbarPos('low h', 'bin')
        high_h = cv2.getTrackbarPos('high h', 'bin')
        low_s = cv2.getTrackbarPos('low s', 'bin')
        high_s = cv2.getTrackbarPos('high s', 'bin')
        low_v = cv2.getTrackbarPos('low v', 'bin')
        high_v = cv2.getTrackbarPos('high v', 'bin')

        lower_yellow = (low_h, low_s, low_v)
        upper_yellow = (high_h, high_s, high_v)
        img_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # img_result = cv2.bitwise_or(img, img, mask=img_mask)
        # img_binary = get_binary(s_channel, s_thresh)

        cv2.imshow("f", frame)
        cv2.imshow("hsv", hsv)
        cv2.imshow("bin", img_mask)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# dir threshold
elif type == 2:
    cv2.createTrackbar('low', 'bin', 0, 40, nothing)
    cv2.createTrackbar('high', 'bin', 0, 40, nothing)

    cv2.setTrackbarPos('low', 'bin', 7)
    cv2.setTrackbarPos('high', 'bin', 13)

    while True:
        low = cv2.getTrackbarPos('low', 'bin')
        high = cv2.getTrackbarPos('high', 'bin')
        dir_thresh = (low/5, high/5)
        dir_binary = abs_sobel_thresh(img, sobel_kernel=3, thresh=dir_thresh)
        cv2.imshow("bin", dir_binary)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# sx_thresh
elif type == 3:
    cv2.createTrackbar('low', 'bin', 0, 255, nothing)
    cv2.createTrackbar('high', 'bin', 0, 255, nothing)

    cv2.setTrackbarPos('low', 'bin', 5)
    cv2.setTrackbarPos('high', 'bin', 40)

    while True:
        low = cv2.getTrackbarPos('low', 'bin')
        high = cv2.getTrackbarPos('high', 'bin')
        sx_thresh = (low, high)
        sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
        cv2.imshow("bin", sxbinary)
        if cv2.waitKey(1) & 0xFF == 27:
            break
# cv2.destroyAllWindows()