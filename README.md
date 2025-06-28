# Artificial Intelligence

![file](https://github.com/user-attachments/assets/0707d875-a1fb-440c-99ba-d1a0292642cf)

> Artificial Intelligence (AI) is a field of computing that develops systems capable of performing tasks that would normally require human intelligence. This includes learning, reasoning, perception, pattern recognition, and decision-making. It uses mathematical models to analyze data and learn from it. The existing models are:

* Machine Learning (ML) = Models that allow computers to learn from data and make predictions or decisions without being explicitly programmed to perform these tasks.
* Neural Networks (NM) = Models inspired by the structure and functioning of the human brain.
* Deep Learning (DL) = Models that use Neural Networks with multiple layers.
* Generative AI = The intelligence, capable of creating new content, such as text, images, music and even videos, based on patterns learned from existing data.
  
### Mind Map

![Imgs](https://github.com/user-attachments/assets/abeade2e-6950-4da2-a21f-61ef039030c8)

### Repository Bibliographic References:
- Aifolks.org (Majority of images and mind maps)
- Perceptrons: An Introduction to Computational Geometry, by Marvin Minsky, Seymour A. Papert

É uma técnica leve de adaptação de modelos de linguagem que não altera os pesos originais do modelo.
✅ Em vez disso, ela aprende pequenos vetores adicionais chamados prompt embeddings que são “injetados” como parte da entrada do modelo.
É o ajuste completo (ou parcial) dos pesos internos do modelo para que ele aprenda melhor uma tarefa específica.

💡 Resumindo:

    Você faz o modelo "esquecer um pouco" o que sabia antes e aprender algo novo.

    Pode ser feito no modelo todo (full fine-tuning) ou só em partes (ex.: LoRA, QLoRA).
    QLoRA vai além: combina LoRA com quantização para economizar ainda mais memória.
    LoRA é uma técnica de fine-tuning eficiente para modelos grandes, criada para reduzir custo e memória durante o treinamento.
