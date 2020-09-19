import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibration import load_calibration, undistort_img

class Lane():
    def __init__(self):
        # was the line detected in the last frame or not
        self.detected = False
        #x values for detected line pixels
        self.cur_fitx = None
        #y values for detected line pixels
        self.cur_fity = None
        # x values of the last N fits of the line
        self.prev_fitx = []
        #polynomial coefficients for the most recent fit
        self.current_poly = [np.array([False])]
        #best polynomial coefficients for the last iteration
        self.prev_poly = [np.array([False])]

        def average_pre_lanes(self):
            tmp = copy(self.prev_fitx)
            tmp.append(self.cur_fitx)
            self.mean_fitx = np.mean(tmp, axis=0)

        def append_fitx(self):
            if len(self.prev_fitx) == N:
                self.prev_fitx.pop(0)
            self.prev_fitx.append(self.mean_fitx)

        def process(self, ploty):
            self.cur_fity = ploty
            self.average_pre_lanes()
            self.append_fitx()
            self.prev_poly = self.current_poly


left_lane = Lane()
right_lane = Lane()

cali_file = '../camera_cal/calibration.p'
mtx, dist = load_calibration(cali_file)

s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

x = [260, 1130, 730, 590]
y = [719, 719, 461, 461]
x_2 = [260, 990, 990, 260]
y_2 = [719, 719, 0, 0]

src = np.float32([[x[0]//2, y[0]//2], [x[1]//2, y[1]//2],[x[2]//2, y[2]//2], [x[3]//2, y[3]//2]])
dst = np.float32([[x_2[0]//2, y_2[0]//2], [x_2[1]//2, y_2[1]//2],[x_2[2]//2, y_2[2]//2], [x_2[3]//2, y_2[3]//2]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

window_margin = 50


def get_binary(channel, thresh):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def find_lines(img):
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    output = np.dstack((img, img, img)) * 255

    mid = int(histogram.shape[0]/2)
    start_left_x = np.argmax(histogram[:mid])
    start_right_x = np.argmax(histogram[mid:]) + mid
    num_windows = 9
    window_h = img.shape[0] // 9

    current_left_x = start_left_x
    current_right_x = start_right_x

    nonzero = img.nonzero()
    nonzero_x = nonzero[1]
    nonzero_y = nonzero[0]

    left_lane_pixels = []
    right_lane_pixels = []

    for window in range(num_windows):
        window_min_y = img.shape[0] - (window + 1) * window_h
        window_max_y = img.shape[0] - window * window_h
        window_left_x_min = current_left_x - window_margin
        window_left_x_max = current_left_x + window_margin
        window_right_x_min = current_right_x - window_margin
        window_right_x_max = current_right_x + window_margin

        cv2.rectangle(output, (window_left_x_min, window_min_y), (window_left_x_max, window_max_y), (0, 255, 0), 2)
        cv2.rectangle(output, (window_right_x_min, window_min_y), (window_right_x_max, window_max_y), (255, 0, 0), 2)

        left_window_inds = ((nonzero_y >= window_min_y) & (nonzero_y <= window_max_y) & (nonzero_x >= window_left_x_min)
                            & (nonzero_x <= window_left_x_max)).nonzero()[0]

        right_window_inds = ((nonzero_y >= window_min_y) & (nonzero_y <= window_max_y) & (nonzero_x >= window_right_x_min)
                            & (nonzero_x <= window_right_x_max)).nonzero()[0]

        left_lane_pixels.append(left_window_inds)
        right_lane_pixels.append(right_window_inds)

        if len(left_window_inds) > 100:
            current_left_x = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > 100:
            current_right_x = np.int(np.mean(nonzero_x[right_window_inds]))

        cy = (window_min_y + window_max_y) // 2
        cv2.circle(output, (current_left_x, cy), 3, (0, 0, 255), -1)
        cv2.circle(output, (current_right_x, cy), 3, (0, 0, 255), -1)
    return output
def find_edges(img, s_thresh=s_thresh):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    s_binary = get_binary(s_channel, s_thresh)

    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)

    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)

    # output mask
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(((sxbinary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (dir_binary == 1)))] = 1

    # add more weights for the s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[((sxbinary == 1) & (s_binary == 1))] = 2

    ave_binary = (combined_binary + c_bi)

    return ave_binary

def warper(img):
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

    return warped

def process_img(img, visualization=False):
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist, (0, 0), fx=1/2, fy=1/2)
    img_binary = find_edges(img_undist)

    cv2.circle(img, (x[0], y[0]), 10, (255, 0, 0), -1)
    cv2.circle(img, (x[1], y[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (x[2], y[2]), 10, (0, 0, 255), -1)
    cv2.circle(img, (x[3], y[3]), 10, (255, 255, 0), -1)

    warped = warper(img_binary)

    output = find_lines(warped)

    cv2.imshow('output', output)




if __name__ == '__main__':
    img = cv2.imread('../test_images/test1.jpg')
    img = process_img(img)
    # cv2.imshow(img)
    cv2.waitKey(0)




