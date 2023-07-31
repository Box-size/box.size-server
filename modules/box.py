from PIL import Image
import numpy as np
import cv2
import detector
import simplifier
import findDot

def calculate_box_size(image : Image, width : int, height : int, focalLength : float):
    
    box, xyxy = detector.detect(image) #박스 감지하고 crop 리턴

    box = simplifier.simplify(box) #배경 지우고 외곽선 검출

    #show(box)

    image = np.array(image)   # image를 cv2에서 사용 가능하게 변환
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    w, h, t = findDot.find(box, image, xyxy)

    print(w,h,t)

    return w, h, t




#TEST & DEBUG
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey()

image = Image.open("modules/images/box9.jpg")
calculate_box_size(image,800,600,1)