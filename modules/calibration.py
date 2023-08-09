import cv2
from PIL import Image
import numpy as np
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError

def chess(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    return cv2.findChessboardCorners(gray, (4,7), flags)

def findRT(image):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*4,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:7].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #타임아웃 20초로 스케쥴 예약
    with ProcessPool(max_workers=6) as pool:
        future = pool.schedule(chess, args=[gray], timeout=20)

    #결과값을 시간 내에 가져왔을 경우
    try:
        ret, corners = future.result()
    except TimeoutError as error:
        ret, corners = False, None
        print("Timeout. skipped.")
    except Exception as error:
        ret, corners = False, None
        print("Function raised %s" % error)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # image = cv2.drawChessboardCorners(image, (4,7), corners2,ret)
        # image = cv2.resize(image,dsize=(800,600))
        # cv2.imshow('img',image)
        # cv2.waitKey(0)

        ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        fx, fy = mtx[0][0], mtx[1][1]
        cx, cy = mtx[0][2], mtx[1][2]
    else:
        return 0, 0, 0, 0, 0, 0
        

    #print(rvec, dist, fx, fy, cx, cy, sep="\n")
    return rvec[0], dist, fx, fy, cx, cy

def findParams(image_PIL):
    '''PIL이미지를 받아 camera parameters를 반환'''

    #cv2이미지로의 변환
    image_cv2 = np.array(image_PIL)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    return findRT(image_cv2)
    

if __name__ == '__main__':
    
    image_PIL = Image.open('modules/images_cali/checkfail.jpg')
    print(findParams(image_PIL))