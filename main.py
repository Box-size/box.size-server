from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import json

app = FastAPI()

@app.post("/api/analyze")
async def get_image_size(image: UploadFile = File(...), f: str = Form(...)):
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
    except Exception:
        return ErrorResponse(400, "잘못된 이미지입니다.")
    
    try:
        f = float(f)
    except ValueError:
        return ErrorResponse(400, "f는 실수 형식이여야 합니다.")

    width, height, tall = 1, 2, 3

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