import numpy as np
import cv2 as cv
import glob

target_camera = 'right'

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard Dimensions
cbrow = 7
cbcol = 7
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob(
    '/home/cat/SGM/camera_calibration/jpg_forcalibration/'+target_camera+'_0.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (cbrow, cbcol), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(1000)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, imgpoints, imgpoints_r, mtx,
                            dist, mtx_r, dist_r, gray.shape[::-1])

# 儲存 mtx, dist
with open('/home/cat/SGM/camera_calibration/mtx_'+target_camera+'.txt', 'w') as fp:
    for list in mtx:
        for data in list:
            fp.write(str(data)+' ')
        fp.write('\n')
with open('/home/cat/SGM/camera_calibration/dist_'+target_camera+'.txt', 'w') as fp:
    for list in dist:
        for data in list:
            fp.write(str(data)+' ')

# # show image

# img = cv.imread('/home/cat/SGM/CSI-Camera/jpg_forcalibration/left_1.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# #  undistort
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)
# cv.imwrite('/home/cat/SGM/CSI-Camera/jpg_forcalibration/calibresult.png', dst)

# cv.destroyAllWindows()
