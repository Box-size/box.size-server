from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from modules import box


app = FastAPI()

@app.post("/api/analyze-1")
async def get_image_size(image: UploadFile = File(...),
                         width: int = Form(...),
                         height: int = Form(...),
                         focalLength: float = Form(...)):
    
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    
    try:
        width, height, tall = box.calculate_box_size(image, width, height, focalLength)
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
async def get_image_size(image: UploadFile = File(...),
                         width: int = Form(...),
                         height: int = Form(...),
                         focalLength: float = Form(...)):
    
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    

    width, height, tall = box.calculate_box_size(image, width, height, focalLength)

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