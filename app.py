from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
from infer import predict_image

app = FastAPI(title="Chest X-Ray Pneumonia Classifier API")

# Set up templates
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate MIME type
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type. Please upload an image."}, status_code=400)

    try:
        # Read and convert uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  

        # Run inference
        prediction, probability = predict_image(image)

        return JSONResponse(content={
            "prediction": prediction,
            "probability": round(probability, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)