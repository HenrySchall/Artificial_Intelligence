# Windows: winget install -e --id UB-Mannheim.TesseractOCR
# MacOS: brew install tesseract
# Linux: sudo apt install tesseract-ocr

# pip install pytesseract
# pip install pillow
# pip install opencv-python

import pytesseract
import numpy as np
import cv2
import requests
from PIL import Image

url = "https://raw.githubusercontent.com/HenrySchall/Databases/main/Artificial%20Intelligence/Computer%20Vision/Reconhecimento%20Óptico%20de%20Caracteres%20(OCR)/Imagens/teste01.jpg"

# Baixar a imagem como bytes
response = requests.get(url)
response.raise_for_status()

# Ler a imagem diretamente em OpenCV (BGR)
img_array = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Mostrar a imagem (fora do Colab)
cv2.imshow("Imagem", img)
cv2.waitKey(0)  # Espera qualquer tecla
cv2.destroyAllWindows()

# Converter para RGB (OpenCV usa BGR por padrão)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Imagem", rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(rgb)
print(texto)