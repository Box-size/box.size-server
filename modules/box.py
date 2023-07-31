from PIL import Image
import numpy as np
import cv2
import detector
import simplifier
import findDot

def calculate_box_size(image : Image, width : int, height : int, focalLength : float):
    
    image = np.array(image)   # image를 cv2에서 사용 가능하게 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    box, xyxy = detector.detect(image) #박스 감지하고 crop 리턴
 
    box = simplifier.simplify(box) #배경 지우고 외곽선 검출

    w, h, t = findDot.find(box, image, xyxy)  #점 찾고 길이 반환   (findDot.main : 과정 이미지 뜨게)

    print(w,h,t)

    return w, h, t




#TEST & DEBUG
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey()

if __name__ == '__main__':
    image = Image.open("modules/images/test1.jpg") # 이미지 테스트
    calculate_box_size(image,800,600,1)