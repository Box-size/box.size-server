import numpy as np
import cv2

def simplify(input):
    '''이미지의 배경을 제거하고 외곽선을 검출'''
    
    
    # def resize_ratio(pic):
    #     size=(256, 256)
    #     #덮어씌울 base_pic 생성
    #     base_pic=np.zeros((size[1],size[0]),np.uint8)
    #     pic1=pic
    #     #원본 사진 비율 보존
    #     h,w=pic1.shape[:2]
    #     ash=size[1]/h
    #     asw=size[0]/w
    #     if asw<ash:
    #         sizeas=(int(w*asw),int(h*asw))
    #     else:
    #         sizeas=(int(w*ash),int(h*ash))
    #     pic1 = cv2.resize(pic1,dsize=sizeas)
    #     base_pic[int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
    #     int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2)]=pic1
    #     return base_pic, sizeas

    #TODO: resize 문제 해결 필요
    #input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) #흑백배경으로 변경
    #input, original_ratio = resize_ratio(input)  #사이즈 변경?
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) #흑백배경으로 변경
    # input, original_ratio = resize_ratio(input)  #사이즈 변경?

    #input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)

    # TODO: 여기서 opencv 써서 배경 제거
    #배경 제거
    # 이미지 크기 얻기
    height, width = input.shape[:2]

    # 관심 영역의 사각형 지정 (중앙 부근에 충분한 공간을 포함)
    margin = 20  # 주변 공간 크기
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)

    print("사각형 지정")

    # 초기 마스크 생성
    mask = np.zeros(input.shape[:2], dtype=np.uint8)

    # GrabCut 실행
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(input, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    print("GrabCut")

    # 결과 마스크에서 가능성 있는 전경/배경 픽셀 선택
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 이미지에 마스크 적용하여 전경 추출
    result = input * mask2[:, :, np.newaxis]

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    nuki = cv2.Canny(result, threshold1=100, threshold2=250)

    print("nuki")

    # 명암비 alpha 0이면 그대로, 양수일수록 명암비가 커진다.
    alpha = 0.5
    input = np.clip((1+alpha) * input - 128 * alpha, 0, 255).astype(np.uint8)
    # 윤곽선만 표시된 이미지
    
    # 배경 제거, 이때 배경은 검정
    # output = remove(input,
    #     bgcolor=[0,0,0,255])
    
    #output = input
    

    # Canny를 통해 외곽선만 검출(threshold는 통상적인 값, 추후 실험을 통해 변경 필요)
    # 이미지, Threshold1: 작을 수록 선이 조금더 길게 나옴, Threshold2: 작을 수록 선이 더 많이 검출됨
    # nuki = cv2.Canny(output, 100, 250)

    # morphology를 위한 kernel 제작 nxn의 kernel로 사각형(MORPH_RECT), 즉 커널이 전부 1로 채워짐
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 커널을 사용해 MORPH_CLOSE -> 커널에 맞게 주변 픽셀 다 선택해서 채우기 때문에 선이 두꺼워진다.
    morphology = cv2.morphologyEx(nuki, cv2.MORPH_CLOSE, kernel)
    print("morphology")
    return morphology, 1

