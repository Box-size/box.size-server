import io, json

from PIL import Image
import numpy as np
import cv2
import simplifier, findDot, calibration, detector
from PIL.ExifTags import TAGS


def rotate_image_with_exif(image):
    
    try:
        # 이미지의 Exif 메타데이터 읽기
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'Orientation':
                    if value == 3:
                        image = image.rotate(180, expand=True)
                    elif value == 6:
                        image = image.rotate(270, expand=True)
                    elif value == 8:
                        image = image.rotate(90, expand=True)
                    break
    except AttributeError:
        pass  # Exif 정보가 없는 경우

    return image
#받은 이미지를 640*640비율에 맞게 자르는 함수
def resize_image(original : cv2, crop : cv2, xyxy):
    #3024*4032가정하고 640*853으로 비율 맞춰서 자르기
    original_resize = cv2.resize(original, (640, 853))
    #자른 후 위 아래 100씩 자르기
    original_resize = original_resize[103:743, 0:640]
    print("original xy : ", original.shape[1], original.shape[0])
    print("original_resize xy : ", original_resize.shape[1], original_resize.shape[0])
    print("xyxy", xyxy)
    #비율
    original_ratio = 640 / original.shape[1]
    xyxy_ratio = original.shape[1] / 640
    #crop된 이미지 비율 맞춰서 resize
    crop_x = int(crop.shape[1] * original_ratio)
    crop_y = int(crop.shape[0] * original_ratio)
    crop_resize = cv2.resize(crop, dsize=(crop_x, crop_y)) 
    print("crop xy : ", crop.shape[1], crop.shape[0])
    print("crop_resize xy : ", crop_resize.shape[1], crop_resize.shape[0])
    #비율에 맞게 xyxy조정
    new_xyxy = [float(xyxy[0][0] / xyxy_ratio), 
                float(xyxy[0][1] / xyxy_ratio - 100),
                float(xyxy[0][2] / xyxy_ratio),
                float(xyxy[0][3] / xyxy_ratio - 100)]
    print(new_xyxy)
    return original_resize, crop_resize, new_xyxy

def calculate_box_size(original : Image, crop : Image, params, xyxy, show=False):

    original = np.array(original)   # image를 cv2에서 사용 가능하게 변환
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    crop = np.array(crop)   # image를 cv2에서 사용 가능하게 변환
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    original, crop, xyxy = resize_image(original, crop, xyxy)
    # box, xyxy = detector.detect(image) #박스 감지하고 crop 리턴

    crop, original_ratio = simplifier.simplify(crop) #배경 지우고, 외곽선 검출

    w, h, t = findDot.find(crop, original, xyxy, original_ratio, params, show=show)  #점 찾고 길이 반환 (show=True 시 과정 이미지 보임)

    print("최종 계산 결과 w, h, t:", w,h,t)

    return w, h, t


def calculate_camera_parameters(image : Image):
    params =  calibration.findParams(image)
    #print(rvec, dist, fx, fy, cx, cy, sep="\n")

    #TODO : 어떤 방식으로 서버에 저장할 지 정하기.
    return params
    # import pickle
    # paramFile = open("modules/params.bin",'wb')
    # pickle.dump(params, paramFile)
    # paramFile.close()




#TEST & DEBUG
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey()

# def main(original_image_data, crop_image_data, params, xyxy):
#     original = Image.open(io.BytesIO(original_image_data))
#     crop = Image.open(io.BytesIO(crop_image_data))
#     original = rotate_image_with_exif(original)
#     crop = rotate_image_with_exif(crop)

#     try:
#         json_params = params.replace("'", '"')
#         params_dict = json.loads(json_params)
#         params_list = [
#             np.array(params_dict["rvec"]),
#             np.array(params_dict["dist"]),
#             params_dict["fx"],
#             params_dict["fy"],
#             params_dict["cx"],
#             params_dict["cy"]
#         ]
#     except Exception:
#         print("params 에러 -> 0 리턴함")
#         return {'width': 0, 'height': 0, 'tall': 0}

#     try:
#         xyxy = json.loads(xyxy)
#     except Exception:
#         print("xyxy 에러 -> 0 리턴함")
#         return {'width': 0, 'height': 0, 'tall': 0}
#     # try:
#     width, height, tall = calculate_box_size(original, crop, params_list, xyxy, show=False)
#     # except Exception:
#     #     width, height, tall = 0, 0, 0
#     result = {'width': width, 'height': height, 'tall': tall}
#     print("최종 결과 주기")
#     return result

if __name__ == '__main__':
    image = Image.open("modules/images_cali/tt2.jpg") # 이미지 테스트
    image_cali = Image.open("modules/images_cali/ch.jpg") #calibration
    image = rotate_image_with_exif(image)
    image_cali = rotate_image_with_exif(image_cali)
    params = calculate_camera_parameters(image_cali)
    crop, xyxy = detector.detect(image)
    print("calculate_camera_parameters 결과:", params)
    calculate_box_size(image,crop, params, xyxy,show=True)