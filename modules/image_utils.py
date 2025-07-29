# Tambahkan import ini di bagian atas main.py
import cv2
import numpy as np

def preprocess_image_for_ocr(image_bytes: bytes):
    """
    Menerapkan teknik preprocessing dasar pada gambar untuk meningkatkan akurasi OCR.
    """
    # 1. Decode byte gambar ke format OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Konversi ke Grayscale (langkah paling umum dan penting)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Terapkan Thresholding untuk membuat gambar menjadi hitam-putih (biner)
    #    Otsu's Binarization secara otomatis menentukan nilai threshold terbaik.
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. (Opsional) Terapkan sedikit blur untuk mengurangi noise sebelum thresholding
    #    Jika gambar sangat 'berbintik', baris ini bisa membantu.
    # binary_image = cv2.medianBlur(gray, 3)
    # _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Kembalikan gambar yang sudah diproses
    return binary_image