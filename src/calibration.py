import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

def calibration_camera(x, y, basepath):
    print('Calibration start')
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(basepath, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (x, y), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # write calibration data with pickle
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    path = os.path.join(basepath, 'calibration.p')
    pickle.dump(dist_pickle, open(path, "wb"))
    print(f"write calibration data in {path}")

    return mtx, dist


def load_calibration(cali_file):
    with open(cali_file, 'rb') as file:
        data = pickle.load(file)
        mtx = data['mtx']
        dist = data['dist']

    return mtx, dist


def undistort_img(img_path, cali_file):
    mtx, dist = load_calibration(cali_file)

    img = cv2.imread(img_path)

    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_undistRGB = cv2.cvtColor(img_undist, cv2.COLOR_BRG2RGB)

    return img_undistRGB


if __name__ == '__main__':
    x, y = 9, 6

    basepath = '../camera_cal/'
    calibration_camera(x, y, basepath)
