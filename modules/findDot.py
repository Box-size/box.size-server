import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

"""
실제 계산
1. crop한 이미지 상의 좌표를 구함
2. 원본 사진으로 좌표 이동
3. 이미지 상의 상자 크기 구하기(이때, 상자 크기 정규화로 3D상에 표현, width=100으로 산정 후 대각선 길이와 이미지 밑부분으로 부터의 거리 곱)
4. 초점거리, 원본사진의 중점, 2D좌표, 이미자 상의 박스 크기를 이용해 임의로 정한 3D좌표를 이용해 카메라 외부 파라미터(rvec, tvec)을 구함
5. Rodrigues(rvec)으로 진짜 카메라 정보를 얻은 후, 실제 월드 좌표계의 박스 한 점과, 카메라의 좌표를 구한후 둘 사이 거리 구함
6. 카메라 초점거리 : 실제 카메라와 물체간 거리 = 이미지 상 박스 크기 : 실제 박스크기 비례식을 이용해 박스 크기 산출
모든 계산 식과 내용은 https://darkpgmr.tistory.com/153 참고
"""

'''
def calculate_box_real_length(edges, original, fx, fy, box):
    """
    윤곽선만 검출한 이미지와 초점거리(fx, fy)로 박스의 실제 가로 세로 높이를 계산한다.
    """
    points = find_points_from_edges_image(edges)
    top, bottom, left_top, left_bottom, right_top, right_bottom = classify_points(points)
    away_x, away_y = min(left_top[0], left_bottom[0]), top[1]
    top, bottom, left_top, left_bottom, right_top, right_bottom = adjust_points(top, bottom, left_top, left_bottom, right_top, right_bottom, away_x, away_y, original, edges, box)
    width, height, tall, img_width = calc_pixel_w_h(top, bottom, left_top, left_bottom, right_top, right_bottom)
    cx, cy = original.shape[1] / 2, original.shape[0] / 2
    retval, rvec, tvec = calculate_parameters(fx, fy, cx, cy, top, bottom, left_top, left_bottom, right_top, right_bottom, width, height, tall)
    print(rvec)
    rvec=np.array([[ 0.03133224],
                    [-0.00468171],
                    [1.55188424]],dtype=np.float32)
    distance = calculate_distance(rvec, tvec, bottom, fx, fy, cx, cy)

    return calculate_real_length(width, height, tall, distance, fx, img_width)
'''

def classify_points(points):
    # y 좌표가 가장 낮은 점을 찾아 맨 위 점으로 설정
    top = min(points, key=lambda p: p[1])
    points.remove(top)

    # y 좌표가 가장 높은 점을 찾아 맨 밑 점으로 설정
    bottom = max(points, key=lambda p: p[1])
    points.remove(bottom)

    # x 좌표가 가장 작은 두 점을 찾아 왼쪽 점으로 설정
    left_points = sorted(points, key=lambda p: p[0])[:2]
    for p in left_points:
        points.remove(p)

    # 남은 두 점은 오른쪽 점으로 설정
    right_points = points

    # 각 왼쪽과 오른쪽의 점들을 높이에 따라 위쪽과 아래쪽으로 분류
    left_top = min(left_points, key=lambda p: p[1])
    left_bottom = max(left_points, key=lambda p: p[1])
    right_top = min(right_points, key=lambda p: p[1])
    right_bottom = max(right_points, key=lambda p: p[1])

    return top, bottom, left_top, left_bottom, right_top, right_bottom


def calc_pixel_w_h(top, bottom, left_top, left_bottom, right_top, right_bottom, diagonal, bottom_ratio):
    """
    이미지 좌표계 상의 가로, 세로, 높이를 추정하는 함수
    """

    top_width = math.sqrt((top[0] - left_top[0])**2 + (top[1] - left_top[1])**2)
    bottom_width = math.sqrt((bottom[0] - right_bottom[0])**2 + (bottom[1] - right_bottom[1])**2)
    top_height = math.sqrt((top[0] - right_top[0])**2 + (top[1] - right_top[1])**2)
    bottom_height = math.sqrt((bottom[0] - left_bottom[0])**2 + (bottom[1] - left_bottom[1])**2)
    tall = (math.sqrt((left_top[0] - left_bottom[0])**2 + (left_top[1] - left_bottom[1])**2) + 
            math.sqrt((right_top[0] - right_bottom[0])**2 + (right_top[1] - right_bottom[1])**2)) / 2
    
    width = bottom_width**2 / top_width
    height = bottom_height**2 / top_height

    h_ratio = height / width
    t_ratio = tall / width
    print("(calc_pixel_w_h)width, h_ratio, t_ratio:", width, h_ratio, t_ratio)
    return 100 * 1, 100 * h_ratio * 1, 100 * t_ratio * 1, width

def calc_diagonal(left_bottom, right_top):
    return math.sqrt((left_bottom[0] - right_top[0])**2 + (left_bottom[1] - right_top[1])**2) / 100

def linear_interpolation(top, bottom, original):
    #top과 bottom을 이은 직선의 기울기
    inclination = (top[1] - bottom[1]) / (top[0] - bottom[0])
    #그 직선이 이미지 바닥에 닿는 점 x좌표
    x_bottom = (original.shape[0] - bottom[1] + (inclination * bottom[0])) / inclination
    #그 직선이 이미지 상단에 닿는 점 x좌표
    x_top = (0 - bottom[1] + (inclination * bottom[0])) / inclination
    
    image_bottom = (x_bottom, original.shape[0])
    image_top = (x_top, 0)
    points = [top, bottom, image_top, image_bottom]

    #바닥~bottom 거리 / 바닥~상단 거리 = 바닥으로부터 거리 비율
    bottom_ratio = math.sqrt((image_bottom[0] - bottom[0])**2 + (image_bottom[1] - bottom[1])**2) / math.sqrt((image_top[0] - image_bottom[0])**2 + (image_top[1] - image_bottom[1])**2)
    top_ratio = math.sqrt((image_top[0] - top[0])**2 + (image_top[1] - top[1])**2) / math.sqrt((image_top[0] - image_bottom[0])**2 + (image_top[1] - image_bottom[1])**2)
    print("top, bottom ratio : ", top_ratio, bottom_ratio)
    return bottom_ratio, points

def find_points_from_edges_image(edges):
    """
    윤곽선만 검출한 이미지에서 최대 점 6개를 가진 도형들의 꼭짓점들을 검출
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hexagon_contours = []

    # 육각형 윤곽선 필터링
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        hexagon_contours.append(approx)
    
    # 근사화된 윤곽선에서 각 꼭지점의 좌표 추출 및 표시
    points = []
    for hexagon in hexagon_contours:
        for point in hexagon:
            x, y = point[0]
            
            points.append((x, y))

    return points


def calculate_parameters(fx, fy, cx, cy, dist, top, bottom, left_top, left_bottom, right_top, right_bottom, width, height, tall):
    """
    외부 파라미터 추정
    """
    #2D 이미지 좌표
    image_points = np.array([[bottom[0], bottom[1]],
                            [left_bottom[0], left_bottom[1]],
                            [right_bottom[0], right_bottom[1]],
                            [left_top[0], left_top[1]], 
                            [right_top[0], right_top[1]],
                            [top[0], top[1]]], 
                            dtype=np.float32)
    #3D 좌표계에 생성한 박스 좌표
    object_points = np.array([[0, 0, 0],
                            [height, 0, 0],
                            [0, width, 0],
                            [width, 0, tall],
                            [0, height, tall],
                            [width, height, tall]],
                            dtype=np.float32)
    cameraMatrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]],
                            dtype=np.float32)

    return cv2.solvePnP(object_points, image_points, cameraMatrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)


def calculate_distance(rvec, tvec, bottom, fx, fy, cx, cy):
    """
    물체와 카메라 사이의 거리 계산
    """
    #3D좌표계의 원점을 실제 월드 좌표계의 점으로 변환
    #구한 R벡터를 원래 회전정보 행렬로 변환
    Ro, _ = cv2.Rodrigues(rvec)

    #3D좌표계의 원점을 구하므로, 카메라 좌표계상 박스 맨 밑점(bottom = 0) 이므로 Pc = tvec
    Pc = tvec

    #픽셀좌표의 정규좌표화
    u = (bottom[0] - cx) / fx
    v = (bottom[1] - cy) / fy

    #정규좌표상의 bottom 좌표
    p_c = np.array([[u], [v], [1]], dtype=np.float32)
    #카메라 원점의 카메라좌표
    C_c = np.array([[0], [0], [0]], dtype=np.float32)
    #월드좌표상의 bottom 좌표
    p_w = Ro.transpose()@(p_c - tvec)
    #월드좌표상의 카메라 좌표
    C_w = Ro.transpose()@(C_c - tvec)

    #지면과 맞닿는 점을 P라 할때, P = C_w + k * (p_w - C_w) 성립,
    #월드좌표계상 지면은 Z = 0이므로 k를 구할 수 있음
    k = -C_w[2]/(p_w[2] - C_w[2])

    #지면좌표상의 bottom 좌표
    ground_x = C_w[0] + k*(p_w[0] - C_w[0])
    ground_y = C_w[1] + k*(p_w[1] - C_w[1])

    #실제 카메라와의 거리
    return math.sqrt((C_w[0] - ground_x)**2 + (C_w[1] - ground_y)**2 + C_w[2]**2)


def calculate_real_length(width, height, tall, distance, fx, img_width, diagonal, bottom_ratio):
    """
    카메라와의 거리를 바탕으로 실제 거리 계산
    """
    #카메라와 거리 : 초점거리 = 실제 박스크기 : 이미지상 박스크기
    real_width = round(img_width * (width/(100*1)) * distance / fx, 2)
    real_height = round(img_width * (height/(100*1)) * distance / fx, 2)
    real_tall = round(img_width * (tall/(100*1)) * distance / fx, 2)

    return real_width, real_height, real_tall

def adjust_points(top, bottom, left_top, left_bottom, right_top, right_bottom, away_y, original_ratio, box):

    points = [top, bottom, left_top, left_bottom, right_top, right_bottom]
    new_points = []
    x_ratio = ((box[0][2] - box[0][0])/original_ratio[0])
    y_ratio = ((box[0][3] - box[0][1])/original_ratio[1])
    for point in points:
        new_points.append((box[0][0] + point[0] * x_ratio, box[0][1] + point[1] * y_ratio - (away_y * y_ratio)))

    return new_points[0], new_points[1], new_points[2], new_points[3], new_points[4], new_points[5]



def find(edges, original, box, original_ratio, params, show=False):

    #카메라 파라미터를 구함
    # import pickle
    # paramFile = open("modules/params.bin",'rb')
    # params = pickle.load(paramFile)   # tuple: (rvec, dist, fx, fy, cx, cy)
    # paramFile.close()

    fx, fy, cx, cy = params[2:]
    dist = params[1]
    rvec = params[0]

    points = find_points_from_edges_image(edges)

    if(show):
        # 찾은 점 시각화
        plt.imshow(edges)
        for x, y in points:
            plt.scatter(x, y, color='red', s=10)
        plt.show()

    if(len(points) > 6 or len(points) <= 0):
        return (0, 0, 0)

    top, bottom, left_top, left_bottom, right_top, right_bottom = classify_points(points)
    #좌표 원본이미지에 맞게 보정

    #좌표 조정
    away_x, away_y = min(left_top[0], left_bottom[0]), top[1]
    top, bottom, left_top, left_bottom, right_top, right_bottom = adjust_points(top, bottom, left_top, left_bottom, right_top, right_bottom, away_y, original_ratio ,box)
    
    if(show):
        new_points = [top, bottom, left_top, left_bottom, right_top, right_bottom]
        plt.imshow(original)
        for x, y in new_points:
            plt.scatter(x, y, color='red', s=10)
        plt.show()

    #상자가 이미지 밑부터 어디까지 떨어졌는지 비율
    bottom_ratio, line = linear_interpolation(top, bottom, original)
    if(show):
        plt.imshow(original)
        for x, y in line:
            plt.scatter(x, y, color='red', s=10)
        plt.show()
    #상자 왼쪽밑 ~ 오른쪽위 거리
    diagonal = calc_diagonal(left_bottom, right_top)

    #이미지 꼭지점 좌표를 토대로 구한 가로, 세로, 높이
    width, height, tall, img_width = calc_pixel_w_h(top, bottom, left_top, left_bottom, right_top, right_bottom, diagonal, bottom_ratio)
    print("이미지 꼭지점 좌표를 토대로 구한 가로, 세로, 높이:", width, height, tall)
    
    #외부 파라미터 추정
    _retval, _rvec, tvec = calculate_parameters(fx, fy, cx, cy, dist, top, bottom, left_top, left_bottom, right_top, right_bottom, width, height, tall)
    #print(rvec, tvec)
  
    if(show):
        # 시각화용 코드
        # 3D 좌표계 상에서 카메라의 위치와 방향 계산
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_position = -np.dot(rotation_matrix.T, tvec)

        # 3D 그래프 생성
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 카메라의 위치와 방향 그리기
        ax.quiver(camera_position[0], camera_position[1], camera_position[2], rvec[0], rvec[1], rvec[2])

        #3D 좌표계에 생성한 박스 좌표
        object_points = np.array([[0, 0, 0],
                                [width, 0, 0],
                                [0, height, 0],
                                [width, 0, tall],
                                [0, height, tall],
                                [width, height, tall]],
                                dtype=np.float32)

        #물체 위치 그리기
        ax.scatter3D(object_points[:, 0], object_points[:, 1], object_points[:, 2])

        # 그래프 표시
        plt.show()

    distance = calculate_distance(rvec, tvec, bottom, fx, fy, cx, cy)
    print("distance:", distance)

    w, h, t = calculate_real_length(width, height, tall, distance, fx, img_width, diagonal, bottom_ratio)
    #TODO: 길이 상수값 나중에 실험 후 확인
    w, h, t = w*2, h*2, t*2
    return (w, h, t)
