from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/analyze")
async def get_image_size(image: UploadFile = File(...), additional_data: str = Form(...)):
    try:
        image = Image.open(image.file) # 이미지 객체 받아오기
        width, height = image.size

        additional_data_dict = json.loads(additional_data)
        return {
            "error": None,
            "data": {
                "width": width,
                "height": height,
                "additional_data": additional_data_dict
                }
            }
    except Exception as e:
        return JSONResponse(status_code=500, content=
                            {
                                "error": 500,
                                "data": "서버 에러입니다."
                            })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)