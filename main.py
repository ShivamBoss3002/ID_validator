import os
import json
import tempfile
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from pdf2image import convert_from_path
from fastapi import FastAPI, UploadFile, Form, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from dotenv import load_dotenv

load_dotenv()

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.keras")
class_names = ['Other', 'Aadhaar Card', 'Pan Card', 'Voter Id']

type_mapping = {
    "aadhaar_card": "Aadhaar Card",
    "pan_card": "Pan Card",
    "voter_id": "Voter Id"
}

MASTER_SECRET_KEY = os.getenv("MASTER_SECRET_KEY")

PROJECT_KEYS_JSON = os.getenv("PROJECT_KEYS_JSON", "{}")
try:
    PROJECT_KEYS = json.loads(PROJECT_KEYS_JSON)
except Exception:
    PROJECT_KEYS = {}


def verify_keys(
    x_secret_key: str = Header(None),
    x_project_id: str = Header(None),
    x_project_key: str = Header(None),
):
    if not MASTER_SECRET_KEY:
        raise HTTPException(status_code=500, detail="MASTER_SECRET_KEY not configured")
    if not x_secret_key or x_secret_key != MASTER_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid Secret Key")
    if not x_project_id or not x_project_key:
        raise HTTPException(status_code=401, detail="Project Id and Project Key are required")
    if x_project_id not in PROJECT_KEYS:
        raise HTTPException(status_code=401, detail=f"Unknown project: {x_project_id}")
    if PROJECT_KEYS.get(x_project_id) != x_project_key:
        raise HTTPException(status_code=401, detail="Invalid Project Key")
    return True


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def predict_array(img_array):
    img_array = tf.cast(img_array, tf.float32)
    img_array = tf.image.resize(img_array, IMG_SIZE)
    img_array = tf.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0]
    class_id = int(np.argmax(pred))
    conf = float(np.max(pred))
    return class_names[class_id], conf, pred


def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img)
    label, conf, _ = predict_array(img_array)
    return {"type": label, "confidence": round(conf, 6)}


def predict_pdf(pdf_path, dpi=200, max_pages=2):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=r"D:\poppler-25.12.0\Library\bin")
    if len(pages) > max_pages:
        return {"error": "Maximum page limit reached", "max_pages": max_pages, "found_pages": len(pages)}

    results = []
    labels = []
    for page in pages:
        page_np = np.array(page)
        if page_np.shape[-1] == 4:
            page_np = page_np[..., :3]
        label, conf, _ = predict_array(page_np)
        results.append((label, conf))
        labels.append(label)

    if len(labels) == 2:
        if labels[0] != labels[1]:
            return {
                "error": "Invalid input",
                "reason": "Pages belong to different document types",
                "page1_type": labels[0],
                "page2_type": labels[1]
            }
        return {"type": labels[0], "confidence": round(float(max(results[0][1], results[1][1])), 6)}

    return {"type": labels[0], "confidence": round(float(results[0][1]), 6)}


def predict_file(path):
    if not os.path.exists(path):
        return {"error": "File not found", "path": path}
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        return predict_image(path)
    if ext == ".pdf":
        return predict_pdf(path)
    return {"error": "Unsupported file type", "ext": ext}



app = FastAPI(
    title="ID Document Validator",
    description="Predict document type and authenticate based on expected type."
)

class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        if request.method == "OPTIONS" and request.headers.get("access-control-request-private-network") == "true":
                response = Response(status_code=200)
                response.headers["Access-Control-Allow-Private-Network"] = "true"
                response.headers["Access-Control-Allow-Origin"] = origin or "*"
                response.headers["Access-Control-Allow-Methods"] = "*"
                response.headers["Access-Control-Allow-Headers"] = "*"
                return response
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

ALLOWED_ORIGINS = ["*"]

app.add_middleware(PrivateNetworkAccessMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(
    file: UploadFile,
    expected_type: str = Form(None),
    auth: bool = Depends(verify_keys)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await file.read())

    try:
        result = predict_file(temp_path)
        if expected_type and expected_type in type_mapping and "type" in result:
            if result["type"] == type_mapping[expected_type]:
                result["Authentication"] = "Valid"
            else:
                result["Authentication"] = "Not valid"
        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/")
def read_root():
    return {"message": "ID Validator API is running"}