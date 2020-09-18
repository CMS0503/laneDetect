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

        
left_lane = Lane()
right_lane = Lane()

cali_file = '../camera_cal/calibration.p'
mtx, dist = load_calibration(cali_file)

s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

x = [260, 1130, 730, 590]
y = [719, 719, 461, 461]
x_2 = [260, 990, 990, 260]
y_2 = [719, 719, 0, 0]

src = np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]])
dst = np.float32([[x_2[0], y_2[0]], [x_2[1], y_2[1]],[x_2[2], y_2[2]], [x_2[3], y_2[3]]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)


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

def process_img(img):
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    # img_undist = cv2.resize(img_undist, (0, 0), fx=1 / 4, fy=1 / 4)
    img_binary = find_edges(img_undist)
    # img = cv2.resize(img, (0, 0), fx=1 / 4, fy=1 / 4)
    cv2.circle(img, (x[0], y[0]), 10, (255, 0, 0), -1)
    cv2.circle(img, (x[1], y[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (x[2], y[2]), 10, (0, 0, 255), -1)
    cv2.circle(img, (x[3], y[3]), 10, (255, 255, 0), -1)

    warped = warper(img)

    binary_sub = np.zeros_like(warped)
    binary_sub[:, int(150):int(-80)] = warped[:, int(150):int(-80)]

    cv2.imshow('img', img)
    cv2.imshow('warped', warped)
    cv2.imshow('bw', binary_sub)


if __name__ == '__main__':
    img = cv2.imread('../test_images/test1.jpg')
    process_img(img)
    cv2.waitKey(0)



