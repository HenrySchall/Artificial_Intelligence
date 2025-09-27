# Reconhecimento Óptico de Caracteres (OCR)

> Reconhecimento Óptico de Caracteres (OCR) é a técnica usada para converter imagens de texto em texto editável e pesquisável, ou seja, extrai algo escrito ou impresso (folha escaneada, foto de um documento ou até uma placa de rua) e transforma em caracteres digitais editáveis. Os textos podem se apresentar de diversas formas, condições, ângulos e resoluções. Sendo assim classifica-se ele em dois tipos de cenários: Controlados ou Naturais.

- Cenários Controlados: O ambiente ou as condições são planejadas, manipuladas e padronizadas para reduzir a influência de fatores externos. Nesse caso, teremos textos facilmente identificáveis, pois não há interferência do fundo no processo de identificação e extração dos caracteres.
- Cenários Naturais: São mais complexos de trabalhar, porque o ambiente ou as condições acontecem de forma espontânea, sem manipulação direta, sendo assim podem ocorrer interferências de fatores externos, como: iluminação, angulo da imagem ou qualidade.

#### Introdução ao Tesseract

> O Tesseract é a engine de Reconhecimento Óptico de Caracteres (OCR) de código aberto, mais usada e conhecida do mercado, desenvolvido originalmente pela HP na década de 1980 e, desde 2006, mantido pelo Google.

![Img1](https://github.com/user-attachments/assets/c7ab8af2-e6d1-4623-99d7-35d8e0942773)

> A imagem (Input) é enviada para o sistema para passar por um pré-processamento (Pre-Processor), onde aplicam as dependências Leptônicas. Leptônica é uma biblioteca de processamento e análise de imagens, desenvolvida em Linguagem C, responsável por fornece um conjunto de funções para manipular imagens em baixo nível, como: leitura e escrita em vários formatos (JPEG, PNG, TIFF), conversões de cor e escala de cinza, operações morfológicas (erosão, dilatação, abertura, fechamento), além de processos de filtragem, redimensionamento e rotação. Em seguida, aplica-se a Engine do Tesseract (Lembrando que a Engine a partir da 4.ª Versão, passou a permitir o uso de redes neurais artificiais, então é possível usar uma base de dados de treinamento ou AI API, para treinar os modelos), que realiza o processo de extração dos caracteres desejados. Nesse ponto, o texto já está digitavel e ajustável (output), todavia pode-se ainda realizar um processo de ajuste de pós-processamento (post-processor), para corrigir anomalias se necessário.

Notas da Versão 
- 1.ª  Versão: Oferecia suporte somente para Inglês
- 2.ª  Versão: Suporte para o português brasileiro, assim como para francês, italiano, alemão, espanhol e holandês
- 3.ª  Versão: Expandiu drasticamente o suporte para incluir idiomas ideográficos (simbólicos) como japonês e chinês, bem como idiomas de escrita da direita para a esquerda, como árabe e hebraico
- 4.ª  Versão (atual): Oferece suporte para mais de 100 idiomas, para caracteres e símbolos


### Fontes:
- https://pypi.org/project/pytesseract/ 
- https://github.com/tesseract-ocr/tesseract
- https://medium.com/@balaajip/optical-character-recognition-99aba2dad314
