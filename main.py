import os
import io
from dotenv import load_dotenv
import pytesseract
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Sesuaikan nama fungsi yang diimpor dari modul logic
from modules.logic import get_structured_summary

# Muat environment variables dari file .env
load_dotenv()

# [Opsional] Sesuaikan path Tesseract jika perlu (untuk Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="API OCR dan Perangkum Surat Dokter",
    description="Ekstrak dan rangkum informasi dari surat dokter menggunakan Tesseract, LangChain, dan Groq.",
    version="1.1.0 (Refactored)"
)

@app.post("/summarize/doctor-note/", tags=["Medical Summarization"])
async def summarize_doctor_note(image: UploadFile = File(...)):
    """
    Menerima gambar surat dokter, melakukan OCR, dan merangkumnya menggunakan AI.
    """
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format file tidak valid. Gunakan PNG, JPG, atau JPEG.")

    try:
        # 1. Proses OCR
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        raw_text = pytesseract.image_to_string(pil_image, lang='eng+ind')

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="OCR tidak dapat mendeteksi teks apa pun pada gambar.")

        # 2. Proses Perangkuman dengan AI (JAUH LEBIH SEDERHANA)
        # Panggil fungsi dari logic.py yang akan mengembalikan dictionary bersih
        summary_json = await get_structured_summary(raw_text)

        # 3. Parsing TIDAK LAGI DIPERLUKAN! Hasilnya sudah berupa JSON (dict)
        return JSONResponse(content={
            "filename": image.filename,
            "summary": summary_json,
            "raw_ocr_text": raw_text.strip()
        })

    except Exception as e:
        # Blok catch-all untuk menangani error dari OCR atau pemanggilan AI
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")


@app.get("/", tags=["General"])
def read_root():
    return {"message": "Selamat datang! Silakan akses /docs untuk mencoba API."}