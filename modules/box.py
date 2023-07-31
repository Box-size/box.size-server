from PIL import Image
import numpy as np
import cv2
import detector
import simplifier
import findDot

def calculate_box_size(image : Image, width : int, height : int, focalLength : float):
    
    image = np.array(image)   # image를 cv2에서 사용 가능하게 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    show(image)

    box, xyxy = detector.detect(image) #박스 감지하고 crop 리턴
    show(box)

    box = simplifier.simplify(box) #배경 지우고 외곽선 검출
    show(box)

    w, h, t = findDot.main(box, image, xyxy)  #findDot.find : 과정 이미지 안뜨는 버전.

    print(w,h,t)

    return w, h, t




#TEST & DEBUG
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey()

image = Image.open("modules/images/test5.jpg") # 이미지 테스트
image.resize((800,600))
calculate_box_size(image,800,600,1)