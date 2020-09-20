import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibration import load_calibration, undistort_img

class Lane():
    def __init__(self):
        # was the line detected in the last frame or not
        self.detected = False
        self.current_fit = None
        self.all_x = None
        self.all_y = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.prev_x = []
        self.radius_of_curvature = None

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


def smoothing(lines, pre_lines=3):
    # collect lines & print average line
    lines = np.squeeze(lines)
    avg_line = np.zeros((720))

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def rad_of_curvature():
    plot_y = left_lane.all_y
    left_x, right_x = left_lane.all_x, right_lane.all_x

    left_x = left_x[::-1]  # Reverse to match top-to-bottom in y
    right_x = right_x[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    width_lanes = abs(right_lane.start_x - left_lane.start_x)
    ym_per_pix = 30 / 360  # meters per pixel in y dimension
    xm_per_pix = 3.7*(360/640) / width_lanes  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(plot_y)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # radius of curvature result
    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad


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

    left_lane_pixels_idxs = []
    right_lane_pixels_idxs = []

    for window in range(num_windows):
        window_min_y = img.shape[0] - (window + 1) * window_h
        window_max_y = img.shape[0] - window * window_h
        window_left_x_min = current_left_x - window_margin
        window_left_x_max = current_left_x + window_margin
        window_right_x_min = current_right_x - window_margin
        window_right_x_max = current_right_x + window_margin

        cv2.rectangle(output, (window_left_x_min, window_min_y), (window_left_x_max, window_max_y), (0, 255, 0), 2)
        cv2.rectangle(output, (window_right_x_min, window_min_y), (window_right_x_max, window_max_y), (0, 255, 0), 2)

        left_window_idxs = ((nonzero_y >= window_min_y) & (nonzero_y <= window_max_y) & (nonzero_x >= window_left_x_min)
                            & (nonzero_x <= window_left_x_max)).nonzero()[0]

        right_window_idxs = ((nonzero_y >= window_min_y) & (nonzero_y <= window_max_y) & (nonzero_x >= window_right_x_min)
                            & (nonzero_x <= window_right_x_max)).nonzero()[0]
        left_lane_pixels_idxs.append(left_window_idxs)
        right_lane_pixels_idxs.append(right_window_idxs)

        if len(left_window_idxs) > 100:
            current_left_x = np.int(np.mean(nonzero_x[left_window_idxs]))
        if len(right_window_idxs) > 100:
            current_right_x = np.int(np.mean(nonzero_x[right_window_idxs]))

        # print('left_window_idxs :', left_window_inds.shape)


        # cy = (window_min_y + window_max_y) // 2
        # cv2.circle(output, (current_left_x, cy), 3, (0, 0, 255), -1)
        # cv2.circle(output, (current_right_x, cy), 3, (0, 0, 255), -1)

    left_lane_pixels_idxs = np.concatenate(left_lane_pixels_idxs)
    right_lane_pixels_idxs = np.concatenate(right_lane_pixels_idxs)

    left_x, left_y = nonzero_x[left_lane_pixels_idxs], nonzero_y[left_lane_pixels_idxs]
    right_x, right_y = nonzero_x[right_lane_pixels_idxs], nonzero_y[right_lane_pixels_idxs]

    output[left_y, left_x] = [255, 0, 0]
    output[right_y, right_x] = [0, 0, 255]
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_plot_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_plot_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]


    left_lane.prev_x.append(left_plot_x)
    right_lane.prev_x.append(right_plot_x)

    if len(left_lane.prev_x) > 10:
        left_avg_line = smoothing(left_lane.prev_x, 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        left_lane.current_fit = left_avg_fit
        left_lane.all_x, left_lane.all_y = left_fit_plot_x, plot_y
    else:
        left_lane.current_fit = left_fit
        left_lane.all_x, left_lane.all_y = left_plot_x, plot_y

    if len(right_lane.prev_x) > 10:
        right_avg_line = smoothing(right_lane.prev_x, 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        right_lane.current_fit = right_avg_fit
        right_lane.all_x, right_lane.all_y = right_fit_plot_x, plot_y
    else:
        right_lane.current_fit = right_fit
        right_lane.all_x, right_lane.all_y = right_plot_x, plot_y

    left_lane.start_x, right_lane.start_x = left_lane.all_x[len(left_lane.all_x) - 1], right_lane.all_x[
        len(right_lane.all_x) - 1]
    left_lane.end_x, right_lane.end_x = left_lane.all_x[0], right_lane.all_x[0]

    left_lane.detected, right_lane.detected = True, True
    # print radius of curvature
    rad_of_curvature()

    return output


def find_edges(img, s_thresh=s_thresh):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float64)
    s_channel = hls[:, :, 2]
    s_binary = get_binary(s_channel, s_thresh)

    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)

    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)

    # output mask
    combined_binary = np.zeros_like(s_channel).astype(np.uint8)
    combined_binary[(((sxbinary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (dir_binary == 1)))] = 255
    # add more weights for the s channel

    return combined_binary

def warper(img):
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    return warped


def draw_lane(img, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
    window_img = np.zeros_like(img)

    left_plot_x, right_plot_x = left_lane.all_x, right_lane.all_x
    plot_y = left_lane.all_y

    left_pts_l = np.array([np.transpose(np.vstack([left_plot_x - window_margin/5, plot_y]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plot_x + window_margin/5, plot_y])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array([np.transpose(np.vstack([right_plot_x - window_margin/5, plot_y]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plot_x + window_margin/5, plot_y])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    pts_left = np.array([np.transpose(np.vstack([left_plot_x+window_margin/5, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x-window_margin/5, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    print(type(window_img[0][0][0]))

    result = cv2.addWeighted(img, 1, window_img, 0.7, 0)
    return result, window_img


def process_img(img, visualization=False):
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)

    img_undist = cv2.resize(img_undist, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    img_binary = find_edges(img_undist)

    cv2.circle(img, (x[0], y[0]), 10, (255, 0, 0), -1)
    cv2.circle(img, (x[1], y[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (x[2], y[2]), 10, (0, 0, 255), -1)
    cv2.circle(img, (x[3], y[3]), 10, (255, 255, 0), -1)

    warped = warper(img_binary)

    output = find_lines(warped)

    # cv2.imshow('output', output)

    return output, img_undist

if __name__ == '__main__':
    img = cv2.imread('../test_images/test1.jpg')
    img, img_undist = process_img(img)

    result_comb, result_color = draw_lane(img)
    rows, cols = result_comb.shape[:2]

    result_color = cv2.warpPerspective(result_color, M_inv, (result_comb.shape[1], result_comb.shape[0]), flags=cv2.INTER_NEAREST)
    comb_result = np.zeros_like(img_undist)

    comb_result[220:rows - 12, 0:cols] = result_color[220:rows - 12, 0:cols]
    # print(comb_result.shape, result_color.shape)
    # comb_result = np.asarray(comb_result, np.uint8)


    result = cv2.addWeighted(img_undist, 0.7, result_color, 0.3, 0)
    # print(type(img_undist[0][0][0]), type(result_color[0][0][0]))

    cv2.imshow('result', result)

    cv2.waitKey(0)




