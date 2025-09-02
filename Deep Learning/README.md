# Deep Learning

> Antes de introduzir Deep Learning e Redes Neurais, é preciso entender o conceito de Regressão Linear Multipla ou Multilinear usada para modelar a relação entre uma variável dependente (y) e duas ou mais variáveis independentes (X1, X2, Xn, ...).

### Equação Geral
$Y = β0 + βX1 + β2X2 + βXn ... + e$

Onde: 
- Y = Variável prevista (dependente).
- $X1$, $X2$, $Xn$ = Variáveis explicativas (independentes)
- $β0$ = intercepto, valor de Y quando todos os X são zero
- $β0$, $βX1$, $β2X2$, $βXn$ = coeficientes que mostram o efeito de cada variável X sobre Y, mantendo as outras constantes.
- e = erro, parte que o modelo não explica.

Já as Redes Neurais são modelos inspirados no funcionamento do cérebro humano, que conseguem aprender padrões complexos à partir de dados. Basicamente estamos falando de um Neurônio Artificial (nó), que recebe entradas (valores) e produz saídas com esses valores.

### Estrutura

<img width="1000" height="350" alt="userlmn_e8761fc5ba29d560be5d9ac8d91adf44" src="https://github.com/user-attachments/assets/0315709b-1656-4093-806c-aed74ebf9158" />

Onde:
- Camada de entrada (Imput Layer): Recebe os dados brutos
- Camadas ocultas (Hidden Layer): Faz as transformações intermediárias, combinando pesos e aplicando funções espécificas (não lineares).
- Camada de saída (Output Layer): Gera o resultado final

> A grande diferença entre a Regressão Linear Multipla, está na camada oculta (Hidden Layer), que é responsável por permitir a não linearidade, já que por meio da função de ativação, decide como a entrada de um neurônio será transformado na sua saída. Outra diferença é a combinação de entradas, gerando uma dinâmica entre elas, transformando em variáveis dinâmicas. Sem esses fatores, uma rede neural seria apenas uma combinação linear de entrada, sendo assim, mesmo com várias camadas ela faria uma simples regressão linear.

### Equação Geral
$$\mathrm{Z} = \sum_{n=1}^{n} Wiβi + B$$

Onde:
- $Wi$ = São os pesos
- B = São os viés
- Z = Função de Ativação, dada por y = $f(z)$
- y = saída do neurônio

Tipos de funções de ativação: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax.

#### Processo de retrocesso 

---
## Referências Biográficas:

### Mind Map
![IMG_2305](https://github.com/user-attachments/assets/a5293786-1ab6-4893-8e7d-9de964db1d37)

## Neural Network
### Referências Biográficas:
- Neural Networks and Deep Learning: A Textbook, by Charu C Aggarwal
- Deep Learning (Adaptive Computation and Machine Learning series),by Ian Goodfellow (Author), Yoshua Bengio (Author), Aaron Courville
- https://www.asimovinstitute.org/neural-network-zoo/
