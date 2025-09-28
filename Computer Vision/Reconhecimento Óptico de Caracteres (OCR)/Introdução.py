# Install Tesseract OCR
#   Windows: winget install -e --id UB-Mannheim.TesseractOCR
#   MacOS: brew install tesseract
#   Linux: sudo apt install tesseract-ocr

# pip install pytesseract
# pip install pillow
# pip install opencv-python
 
import pytesseract
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO

#########################
### Imagem 1 (OpenCV) ###
#########################

#url = "https://raw.githubusercontent.com/HenrySchall/Databases/main/Artificial%20Intelligence/Computer%20Vision/Reconhecimento%20Óptico%20de%20Caracteres%20(OCR)/Imagens/teste01.jpg"
url = "https://drive.google.com/uc?export=download&id=1CXcIsaSjyQBDL53_6UvZH7-aSd-6NZwj"

# Baixar a imagem como bytes
response = requests.get(url)
response.raise_for_status()

# Ler a imagem diretamente em OpenCV (Usa padrão BGR)
img_array = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Mostrar a imagem
cv2.imshow("Imagem", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Converter para RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(rgb)
print(texto)

#########################
### Imagem 2 (Pillow) ###
#########################

url2 = "https://drive.google.com/uc?export=download&id=1QC4npS_HDYcxJsZuXMnbW1BwhC7py9vt"

response = requests.get(url2)
response.raise_for_status()

# Ler a imagem diretamente em Pillow (Usa padrão RGB)
img = Image.open(BytesIO(response.content))
img.show()

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(img)
print(texto)

# Observe que o Tesseract reconhece automaticamente o idioma do texto.
# Caso queira definir o idioma, baixe o pacote de idiomas desejado, usando o parâmetro 'lang':

texto = pytesseract.image_to_string(rgb, lang='por') 
print(texto) 

# Se não funcionar, verifique se o idioma foi instalado corretamente:
# No cmd, rode o comando: "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
# Caso não tenha o idioma, baixe o arquivo .traineddata do idioma desejado (link: https://github.com/tesseract-ocr/tessdata)
# e coloque na pasta "C:/Program Files/Tesseract-OCR/tessdata"

texto = pytesseract.image_to_string(rgb, lang='por') 
print(texto) 

####################################
### Page Segmentation Mode (PSM) ###
####################################

url3 = "https://drive.google.com/uc?export=download&id=1MHsLmCXfq2eXLE0FvuqK7aoMFn8xLw5k"

response = requests.get(url3)
response.raise_for_status() 

img = Image.open(BytesIO(response.content))
img.show()


