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

#######################################
### Introdução ao OCR com Tesseract ###
#######################################

#########################
### Imagem 1 (OpenCV) ###
#########################

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
# Para obter ajuda No cmd, rode o comando: "C:\Program Files\Tesseract-OCR\tesseract.exe" --help-nome_do_comando

texto = pytesseract.image_to_string(rgb, lang='por') 
print(texto) 

####################################
### Page Segmentation Mode (PSM) ###
####################################

#################
### Imagem 1 ####
#################

url3 = "https://drive.google.com/uc?export=download&id=1sj15Be8h9josm97IKr3_g5vi9ZkilPUl"

response = requests.get(url3)
response.raise_for_status()  

img = Image.open(BytesIO(response.content))
img.show()

# Sem PSM 6 (Padrão: 3 - Assume uma página com vários blocos de texto)
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
texto = pytesseract.image_to_string(img, lang='por')
print(texto)

# Com PSM 6 (Assume um único bloco uniforme de texto)
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
config_tesseract = '--psm 6'
texto = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
print(texto)

################
### Imagem 2 ###
################

url4 = "https://drive.google.com/uc?export=download&id=1qHKUrP17MyZD8PrplEPRGBo1MhXYC--h"

response = requests.get(url4)
response.raise_for_status()  

img2 = Image.open(BytesIO(response.content))
img2.show()

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
config_tesseract = '--psm 7'
texto = pytesseract.image_to_string(img2, lang='por', config=config_tesseract)
print(texto)