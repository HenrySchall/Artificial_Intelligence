# Reconhecimento Óptico de Caracteres (OCR)

> Reconhecimento Óptico de Caracteres (OCR) é a técnica usada para converter imagens de texto em texto editável e pesquisável, ou seja, extrai algo escrito ou impresso (folha escaneada, foto de um documento ou até uma placa de rua) e transforma em caracteres digitais editáveis. Os textos podem se apresentar de diversas formas, condições, ângulos e resoluções. Sendo assim classifica-se ele em dois tipos de cenários: Controlados ou Naturais.

- Cenários Controlados: O ambiente ou as condições são planejadas, manipuladas e padronizadas para reduzir a influência de fatores externos. Nesse caso, teremos textos facilmente identificáveis, pois não há interferência do fundo no processo de identificação e extração dos caracteres.
- Cenários Naturais: São mais complexos de trabalhar, porque o ambiente ou as condições acontecem de forma espontânea, sem manipulação direta, sendo assim podem ocorrer interferências de fatores externos, como: iluminação, angulo da imagem ou qualidade.

#### Introdução ao Tesseract

> O Tesseract é a engine de Reconhecimento Óptico de Caracteres (OCR) de código aberto, mais usada e conhecida do mercado, desenvolvido originalmente pela HP na década de 1980 e, desde 2006, mantido pelo Google.

![Img1](https://github.com/user-attachments/assets/c7ab8af2-e6d1-4623-99d7-35d8e0942773)

> A imagem (Input) é enviada para o sistema para passar por um pré-processamento (Pre-Processor), onde aplicam as dependências Leptônicas. Leptônica é uma biblioteca de processamento e análise de imagens, desenvolvida em Linguagem C, responsável por fornece um conjunto de funções para manipular imagens em baixo nível, como: leitura e escrita em vários formatos (JPEG, PNG, TIFF), conversões de cor e escala de cinza, operações morfológicas (erosão, dilatação, abertura, fechamento), além de processos de filtragem, redimensionamento e rotação. Em seguida, aplica-se a Engine do Tesseract (Lembrando que a Engine a partir da 4.ª Versão, passou a permitir o uso de redes neurais artificiais, então é possível usar uma base de dados de treinamento ou AI API, para treinar os modelos), que realiza o processo de extração dos caracteres desejados, em outras palavras, ele vai percorrer linha por linha identificando e extraindo cara letra, de cada palavra. Nesse ponto, o texto já está digitavel e ajustável (output), todavia pode-se ainda realizar um processo de ajuste de pós-processamento (post-processor), para corrigir anomalias se necessário.

Observação: O Tesseract ainda tem uma função também muito importante para detectar a oritentação do texto na imagem, assim como o alfabeto em que ele é escrito, essa função é chamada de OSD (Orientation and Script Detection), muito utilizado para pré-processamento.

Notas da Versão 
- 1.ª  Versão: Oferecia suporte somente para Inglês
- 2.ª  Versão: Suporte para o português brasileiro, assim como para francês, italiano, alemão, espanhol e holandês
- 3.ª  Versão: Expandiu drasticamente o suporte para incluir idiomas ideográficos (simbólicos) como japonês e chinês, bem como idiomas de escrita da direita para a esquerda, como árabe e hebraico
- 4.ª  Versão (atual): Oferece suporte para mais de 100 idiomas, para caracteres e símbolos

Outras opções para OCR
- EasyOCR é uma biblioteca que tem como proposta ser uma solução eficiente e prática que, oferece suporte para mais de 80 linguagens (até o presente momento). Sua acurácia bate de frente com os resultados do Tesseract, e sua detecção de textos nativa aplicado em cenários menos controlados demonstrou superar todas as outras soluções.
- OCRopus solução Open Source que permite fácil avaliação e reutilização dos componentes de OCR por pesquisadores e empresas. É uma coleção de programas de análise de documentos, não um sistema OCR pronto para uso. Para aplicá-lo aos seus documentos, pode ser necessário fazer um pré processamento de imagem e, possivelmente, também treinar novos modelos.
- Ocular funciona melhor emdocumentos impressos em uma impressora manual, incluindo aqueles escritos em vários idiomas. Ele opera usando a linha de comando.
- SwiftOCR biblioteca simples e rápida escrita na linguagem Swift. Usa redes neurais para reconhecimento de imagem.

### Fontes:
- https://pypi.org/project/pytesseract/ 
- https://github.com/tesseract-ocr/tesseract
- https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/33418.pdf
- https://medium.com/@balaajip/optical-character-recognition-99aba2dad314
