import cv2
from PIL import Image
import numpy as np

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
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (4,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #image = cv2.drawChessboardCorners(image, (4,7), corners2,ret)
        #image = cv2.resize(image,dsize=(800,600))
        #cv2.imshow('img',image)
        #cv2.waitKey(0)

        ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        fx, fy = mtx[0][0], mtx[1][1]
        cx, cy = mtx[0][2], mtx[1][2]
    else:
        #TODO : 실패하였을 경우 작성하기.
        pass

    #print(rvec, dist, fx, fy, cx, cy, sep="\n")
    return rvec, dist, fx, fy, cx, cy

def findParams(image_PIL):
    '''PIL이미지를 받아 camera parameters를 반환'''

    #cv2이미지로의 변환
    image_cv2 = np.array(image_PIL)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    return findRT(image_cv2)
    

if __name__ == '__main__':
    
    image_PIL = Image.open('modules/images_cali/check2.jpg')
    findParams(image_PIL)