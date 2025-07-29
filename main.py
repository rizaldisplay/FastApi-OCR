import os
import io
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import cv2  # Import baru
import numpy as np # Import baru

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Adjust the name of the function imported from the logic module.
from modules.logic import get_structured_summary

# Load environment variables from .env file
load_dotenv()

# [Optional] Adjust Tesseract path if needed (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# FastAPI application initialization
app = FastAPI(
    title="API OCR dan Perangkum Surat Dokter",
    description="Ekstrak dan rangkum informasi dari surat dokter menggunakan Tesseract, LangChain, dan Groq.",
    version="1.1.0 (Refactored)"
)

# Letakkan fungsi preprocessing di sini
def preprocess_image_for_ocr(image_bytes: bytes):
    # (Salin fungsi dari Langkah 2 di sini)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

@app.post("/summarize/doctor-note/", tags=["Medical Summarization"])
async def summarize_doctor_note(image: UploadFile = File(...)):
    """
    Menerima gambar surat dokter, melakukan OCR, dan merangkumnya menggunakan AI.
    """
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format file tidak valid. Gunakan PNG, JPG, atau JPEG.")

    try:
        # 1. OCR Processing
        contents = await image.read()
        # 2. Lakukan Preprocessing menggunakan OpenCV
        processed_image = preprocess_image_for_ocr(contents)

        # 3. Lakukan OCR pada gambar yang sudah diproses
        #    Tesseract bekerja lebih baik dengan gambar yang sudah diproses OpenCV
        raw_text = pytesseract.image_to_string(processed_image, lang='eng+ind')

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="OCR tidak dapat mendeteksi teks apa pun pada gambar.")

        # 4. Proses Perangkuman AI (tidak berubah)
        summary_json = await get_structured_summary(raw_text)

        # 5. Kembalikan hasil
        return JSONResponse(content={
            "filename": image.filename,
            "summary": summary_json,
            "raw_ocr_text": raw_text.strip()
        })

    except Exception as e:
        # Catch-all block to handle errors from OCR or AI calls
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")


@app.get("/", tags=["General"])
def read_root():
    return {"message": "Selamat datang! Silakan akses /docs untuk mencoba API."}