import cv2
import numpy as np
def findRT():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    fname = 'firstTest/check2.jpg'
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (4,7),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (4,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(rvec, tvec)
        fx, fy = mtx[0][0], mtx[1][1]
        cx, cy = mtx[0][2], mtx[1][2]
        print(dist)
    return tvec, dist, fx, fy, cx, cy

def main():
    rvec, tvec = findRT()

if __name__ == '__main__':
    main()