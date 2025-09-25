#####################
### Configuration ###
#####################

# My License Keys
# https://drive.google.com/file/d/1Cu5iQHkbF47FLSXpTH-l75ArX9KBDByd/view?usp=sharing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(20)
os.environ["HF_TOKEN"] = "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"


!pip install Pillow==9.1.0 # necessário após atualização no módulo Pillow carregado pelo Colab
# Após executar, clique no botão [Restart Runtime] que vai aparecer no output dessa célula, logo abaixo. Em seguida, pode continuar executando normalmente o restante do código

!sudo apt install tesseract-ocr
!pip install pytesseract

import pytesseract
import numpy as np
import cv2 # OpenCV
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/teste01.jpg')
cv2_imshow(img) # BGR (RGB)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

texto = pytesseract.image_to_string(rgb)

print(texto)

img = cv2.imread('teste02.jpg')
cv2_imshow(img)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

texto = pytesseract.image_to_string(rgb)
print(texto) # resumé, fiancé, déjà vu

!tesseract --list-langs

!apt-get install tesseract-ocr-por

!tesseract --list-langs
