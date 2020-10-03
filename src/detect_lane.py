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
        self.curve_info = None

left_lane = Lane()
right_lane = Lane()

cali_file = '../camera_cal/calibration.p'
mtx, dist = load_calibration(cali_file)

s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (80, 255), (5, 40), (0.7, 1.3), (30, 100), (200, 255)

lower_yellow = (15, 60, 113)
upper_yellow = (33, 255, 255)

lower_white = (0, 0, 180)
upper_white = (255, 50, 255)

x = [200, 1100, 700, 580]
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
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
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
    lines = np.squeeze(lines)
    avg_line = np.zeros((360))

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def rad_of_curvature():
    plot_y = left_lane.all_y
    left_x, right_x = left_lane.all_x, right_lane.all_x

    left_x = left_x[::-1]
    right_x = right_x[::-1]

    width_lanes = abs(right_lane.start_x - left_lane.start_x)
    ym_per_pix = 30 / 360
    xm_per_pix = 3.7*(360/640) / width_lanes

    y_eval = np.max(plot_y)

    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad


def find_lines(img):
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    output = np.dstack((img, img, img)) * 255

    histogram = histogram / 255
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

        # lane window
        cv2.rectangle(output, (window_left_x_min, window_min_y), (window_left_x_max, window_max_y), (0, 255, 0), 2)
        cv2.rectangle(output, (window_right_x_min, window_min_y), (window_right_x_max, window_max_y), (0, 255, 0), 2)

        # lane pixels
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

        cy = (window_min_y + window_max_y) // 2
        cv2.circle(output, (current_left_x, cy), 3, (0, 0, 255), -1)
        cv2.circle(output, (current_right_x, cy), 3, (0, 0, 255), -1)

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

    # visualization Lane
    # print(right_plot_x)
    # print(plot_y)
    # output[np.int_(plot_y)-2, np.int_(left_plot_x)-2] = [255, 255, 255]
    # output[np.int_(plot_y)-10, np.int_(right_plot_x)-10] = [255, 255, 255]
    cv2.imshow("asd", output)

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
    rad_of_curvature()

    return output


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = (255, 255, 255)
    else:
        color = (255)

    cv2.fillPoly(mask, vertices, color)

    roi_img = cv2.bitwise_and(img, mask)
    return roi_img


def find_edges(img, s_thresh=s_thresh):
    img = np.copy(img)

    # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    # s_channel = hls[:, :, 2]
    # s_binary = get_binary(s_channel, s_thresh)

    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)

    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # cv2.imshow('yellow_mask', yellow_mask)
    # cv2.imshow('white_mask', white_mask)
    # cv2.imshow('sxbinary', sxbinary)

    yw_binary = cv2.bitwise_or(yellow_mask, white_mask)
    # cv2.imshow('yw_binary', yw_binary)

    # output mask
    combined_binary = np.zeros_like(yellow_mask).astype(np.uint8)
    combined_binary[(((sxbinary == 255) & (dir_binary == 1)) | ((yw_binary == 255) & (dir_binary == 1)))] = 255
    # combined_binary[((sxbinary == 255) & (yw_binary == 255))] = 255
    # cv2.imshow('combined_binary', combined_binary)

    # ROI
    height, width = combined_binary.shape[:2]
    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    roi = region_of_interest(combined_binary, vertices)

    return roi, yw_binary

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

    result = cv2.addWeighted(img, 1, window_img, 0.7, 0)
    return result, window_img


def process_img(img, visualization=False):
    # Calibration
    img_undist_ori = cv2.undistort(img, mtx, dist, None, mtx)
    img_undist = cv2.resize(img_undist_ori, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

    # ROI
    height, width = img_undist.shape[:2]
    vertices = np.array(
        [[(0, height - 20), (0, 0), (width, 0), (width, height-20)]],
        dtype=np.int32)
    img_undist = region_of_interest(img_undist, vertices)
    cv2.imshow('img_undist',img_undist)
    img_binary, yw_binary = find_edges(img_undist)

    warped_ori = warper(img_undist)

    warped = warper(img_binary)

    # cv2.imshow('warped', warped)
    # cv2.imshow('img_binary', img_binary)

    output = find_lines(warped)

    if visualization is True:
        cv2.circle(img_undist, (x[0]//2, y[0]//2), 10, (255, 0, 0), -1)
        cv2.circle(img_undist, (x[1]//2, y[1]//2), 10, (0, 255, 0), -1)
        cv2.circle(img_undist, (x[2]//2, y[2]//2), 10, (0, 0, 255), -1)
        cv2.circle(img_undist, (x[3]//2, y[3]//2), 10, (255, 255, 0), -1)

        cv2.circle(img_undist, (x_2[0]//2, y_2[0]//2), 10, (200, 0, 0), -1)
        cv2.circle(img_undist, (x_2[1]//2, y_2[1]//2), 10, (0, 200, 0), -1)
        cv2.circle(img_undist, (x_2[2]//2, y_2[2]//2), 10, (0, 0, 200), -1)
        cv2.circle(img_undist, (x_2[3]//2, y_2[3]//2), 10, (200, 200, 0), -1)

        cv2.imshow('img_undist', img_undist)
        # cv2.imshow('warped', warped_ori)
    # cv2.imshow('combined_binary', img_binary)
    # cv2.imshow('yw_binary', yw_binary)
    return output, img_undist, warped_ori


def create_info_image(img_binary, img_window, img_result, img_warped):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_binary = cv2.resize(img_binary, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    img_window = cv2.resize(img_window, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    img_result = cv2.resize(img_result, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    img_warped = cv2.resize(img_warped, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

    w = img_binary.shape[1]
    h = img_binary.shape[0]

    img[40:40+h, 20:20+w, :] = img_binary
    img[40:40+h, 20*2+w:20*2+w*2, :] = img_result
    img[40+h+70:40+h*2+70, 20:20 + w, :] = img_warped
    img[40 + h + 80:40 + h * 2 + 80, 30 + w:30 + w*2, :] = img_window
    # img[40:40 + h, 20:20 + w, :] = binary_img

    font = cv2.FONT_HERSHEY_SIMPLEX

    origin = "Origin"
    DL = "Detect Lane"
    BV = "Bird View"
    SL = "Sliding Window"

    origin_size = cv2.getTextSize(origin, font, 1, 2)[0]
    DL_size = cv2.getTextSize(DL, font, 1, 2)[0]
    BV_size = cv2.getTextSize(BV, font, 1, 2)[0]
    SL_size = cv2.getTextSize(SL, font, 1, 2)[0]

    cv2.putText(img, origin, ((w - origin_size[0]) // 2, 20 + h + 60), font, 1, (255 , 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, DL, (20*2+w+(w - DL_size[0]) // 2, 20 + h + 60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, BV, ((w - BV_size[0]) // 2, 20 + h*2 + 140), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, SL, (20*2+w+(w - SL_size[0]) // 2, 20 + h*2 + 140), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    curvature = (left_lane.radius_of_curvature + right_lane.radius_of_curvature) / 2
    direction = ((left_lane.end_x - left_lane.start_x) + (right_lane.end_x - right_lane.start_x)) / 2

    if curvature > 4500 and abs(direction) < 100:
        curve_info = 'No Curve'
    elif curvature <= 4500 and direction < - 40:
        curve_info = 'Left Curve'
    elif curvature <= 4500 and direction > 40:
        curve_info = 'Right Curve'
    else:
        if left_lane.curve_info != None:
            curve_info = left_lane.curve_info
        else:
            curve_info = 'None'

    cv2.putText(img, curve_info, (20*2+w+20, 80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, str(int(curvature)), (105 + w * 2, 80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, str(int(direction)), (105 + w * 2, 120), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return img, curve_info


if __name__ == '__main__':
    type = 1  # 0:image 1:video
    if type == 0:
        image = 'test9'
        img = cv2.imread(f'../test_images/{image}.jpg')
        process_img(img)
        # img, img_undist, _ = process_img(img)
        #
        # result_comb, result_color = draw_lane(img)
        # rows, cols = result_comb.shape[:2]
        #
        # result_color = cv2.warpPerspective(result_color, M_inv, (result_comb.shape[1], result_comb.shape[0]), flags=cv2.INTER_NEAREST)
        # comb_result = np.zeros_like(img_undist)
        #
        # comb_result[220:rows - 12, 0:cols] = result_color[220:rows - 12, 0:cols]
        #
        # result = cv2.addWeighted(img_undist, 1, result_color, 0.3, 0)
        #
        # cv2.imshow('result', result)
        cv2.waitKey(0)

    else:
        video = "challenge"
        cap = cv2.VideoCapture(f'../{video}_video.mp4')

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter(f'../{video}.avi', fourcc, 30, (1280, 720))
        while(cap.isOpened()):
            try:
                _, frame = cap.read()
                process_img(frame)
                img_window, img_undist, img_warped = process_img(frame)

                result_comb, result_color = draw_lane(img_window)
                rows, cols = result_comb.shape[:2]

                result_color = cv2.warpPerspective(result_color, M_inv, (result_comb.shape[1], result_comb.shape[0]),
                                                   flags=cv2.INTER_NEAREST)
                comb_result = np.zeros_like(img_undist)
                comb_result[220:rows - 12, 0:cols] = result_color[220:rows - 12, 0:cols]

                result = cv2.addWeighted(img_undist, 0.7, result_color, 0.3, 0)
                cv2.imshow('result', result)
                img_info, curve_info = create_info_image(img_undist, img_window, result, img_warped)

                left_lane.curve_info = curve_info

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(e)
                break

        cap.release()
        cv2.destroyAllWindows()