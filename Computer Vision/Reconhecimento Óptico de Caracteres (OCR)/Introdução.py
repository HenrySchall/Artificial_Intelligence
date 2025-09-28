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

################
### Imagem 1 ###
################

url = "https://raw.githubusercontent.com/HenrySchall/Databases/main/Artificial%20Intelligence/Computer%20Vision/Reconhecimento%20Óptico%20de%20Caracteres%20(OCR)/Imagens/teste02.jpg"

# Baixar a imagem como bytes
response = requests.get(url)
response.raise_for_status()

# Ler a imagem diretamente em OpenCV (BGR)
img_array = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Mostrar a imagem
cv2.imshow("Imagem", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Converter para RGB (OpenCV usa BGR por padrão)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(rgb)
print(texto)

################
### Imagem 2 ###
################

url2 = "https://raw.githubusercontent.com/HenrySchall/Databases/main/Artificial%20Intelligence/Computer%20Vision/Reconhecimento%20Óptico%20de%20Caracteres%20(OCR)/Imagens/teste02.jpg"

response = requests.get(url2)
response.raise_for_status()

img_array = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("Imagem", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(rgb)
print(texto) 

# Observe que o Tesseract reconhece automaticamente o idioma do texto.
# Caso queira definir o idioma, baixe o pacote de idiomas desejado, usando o parâmetro 'lang':

texto = pytesseract.image_to_string(rgb, lang='por') 
print(texto) 

# Se não funcionar, verifique se o idioma foi instalado corretamente:
# No cmd, rode o comando: "C:/Program Files/Tesseract-OCR/tesseract.exe" --list-langs
# Caso não tenha o idioma, baixe o arquivo .traineddata do idioma desejado (link: https://github.com/tesseract-ocr/tessdata)
# e coloque na pasta "C:/Program Files/Tesseract-OCR/tessdata"
