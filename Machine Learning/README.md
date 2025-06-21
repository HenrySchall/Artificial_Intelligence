# Machine Learning

#### Types of Learning
- Supervised = In supervised learning, the model is trained with data that already has the associated responses (labels). In other words, you provide the model with input examples and the respective desired outputs
- Unsupervised = The model works with data that does not have labels. The goal is to discover hidden patterns or structures (classification) within the data without knowing in advance what the correct outputs are, that is, "no output"
- Reinforcement Learning = In reinforcement learning, the model learns by trial and error, that is, it makes decisions in a dynamic environment, and learns from the feedback it receives, indicating errors, but not providing solutions

### Mind Map

![IMG_2305](https://github.com/user-attachments/assets/a5293786-1ab6-4893-8e7d-9de964db1d37)

### Repository Bibliographic References:
- Introduction to Machine Learning (Adaptive Computation and Machine Learning), Ethem Alpaydin
- Foundations of Machine Learning, Mehryar Mohri, Afshin Rostamizadeh, Ameet Talwalkar


Machine Learning
- https://app.datacamp.com/learn/courses/machine-learning-for-time-series-data-in-python
- https://medium.com/equals-lab/uma-breve-introdu%C3%A7%C3%A3o-ao-algoritmo-de-machine-learning-gradient-boosting-utilizando-a-biblioteca-311285783099 
- https://www.coursera.org/professional-certificates/ibm-machine-learning 
- https://www.coursera.org/learn/fundamentals-machine-learning-in-finance 




O pickle é um módulo em Python usado para serializar e desserializar objetos Python, ou seja, converter objetos Python em uma sequência de bytes para que possam ser salvos em um arquivo ou transmitidos pela rede e, posteriormente, reconstruí-los de volta no formato original.

O que o pickle faz:
Serialização (Pickling): A conversão de um objeto Python (como listas, dicionários, classes, etc.) em uma sequência de bytes. Isso permite salvar o objeto em um arquivo ou em memória.

Desserialização (Unpickling): O processo reverso, onde os bytes salvos são convertidos de volta ao objeto original, permitindo que você recupere e use o objeto salvo.

Quando usar pickle:
Salvar modelos de machine learning: Depois de treinar um modelo (como um modelo de regressão, classificação, etc.), você pode usar pickle para salvar o modelo em um arquivo e carregá-lo mais tarde sem precisar treinar novamente.

Salvar o estado de um programa: Em aplicações que requerem salvar o estado atual de objetos Python para posterior recuperação.

Transmissão de dados complexos: Se você precisa enviar ou armazenar objetos Python complexos, pickle pode ser uma forma eficiente de fazer isso.

Considerações de segurança:
Cuidado com a desserialização de dados não confiáveis: Não é seguro carregar objetos usando pickle de fontes não confiáveis, pois isso pode executar código malicioso durante o processo de desserialização.

Padronizacao e indicada quando se tem mais outlier na base de dados, e normalizacao 
Nao supervisionada descobre os padroes
