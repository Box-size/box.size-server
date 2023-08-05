from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from modules import box
import json
import numpy as np

app = FastAPI()

@app.post("/api/test")
async def get_image_size_test(image: UploadFile = File(...)):
    
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    
    try:
        rvec, dist, fx, fy, cx, cy = box.calculate_camera_parameters(image)
    except Exception:
        return ErrorResponse(400, "인식되지않았습니다. 다시 시도해주세요.")

    params = {
        "rvec": rvec.tolist(),
        "dist": dist.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }

    # print(json.dumps(params))
# rvec, dist, fx, fy, cx, cy
    return {
            "status": 200,
            "response": {
                "params" : params
            },
            "errorMessage": None
        }

def ErrorResponse(status, message):
    return JSONResponse(status_code=status, content=
                            {
                                "status": status,
                                "response": None,
                                "errorMessage": message
                            })

@app.post("/api/analyze-1")
async def get_image_size1(image: UploadFile = File(...), params : str = File(...)):
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    try:    
        params_dict = json.loads(params)
        params_list = [np.array(params_dict["rvec"]), np.array(params_dict["dist"]), params_dict["fx"], params_dict["fy"], params_dict["cx"], params_dict["cy"]]
    except Exception:
        return ErrorResponse(400, "/api/test 결과의 params를 그대로 돌려주세요")

    try:
        width, height, tall = box.calculate_box_size(image, params_list)
    except Exception:
        width, height, tall = 0, 0, 0

    return {
        "status": 200,
        "response": {
            "width": width,
            "height": height,
            "tall": tall
            },
        "errorMessage": None
        }

@app.post("/api/analyze-2")
async def get_image_size2(image: UploadFile = File(...)):
    
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    

    try:
        width, height, tall = box.calculate_box_size(image)
    except Exception:
        width, height, tall = 0, 0, 0

    return {
        "status": 200,
        "response": {
            "width": width,
            "height": height,
            "tall": tall
            },
        "errorMessage": None
        }

def ErrorResponse(status, message):
    return JSONResponse(status_code=status, content=
                            {
                                "status": status,
                                "response": None,
                                "errorMessage": message
                            })
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)