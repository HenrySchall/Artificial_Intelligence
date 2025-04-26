# Foundation Models

> Foundation Models são modelos de inteligência artificial (AI) de grande escala, baseados em técnicas de Deep Learning (Redes Neurais Profundas) de aprendizado não-supervisionado, ou seja, sem qualquer instrução humana. Eles são utilizados em enormes volumes de dados para criar conexões e captura as relações complexas entre os dados, para depois podem ser adaptados para tarefas específicas. O termo foundation foi popularizado pela Universidade de Stanford em 2021, devido ao fato que eles funcionam como base para construção de outras aplicações, em vez de treinarem um modelo do zero para cada tipo problema. Os Large Language Models (LLM's) são uma classe dos Foundation Models, especificos para linguagem, portando focam em tarefas relacionadas a texto, nesse contexto que surge os conceitos de Prompt Engineering (prática de criar instruções (prompts) inteligentes para controlar e guiar o comportamento de modelos de linguagem) e Natural Language Processing (NLP) (prática de ensinar computadores a entender, interpretar, gerar e interagir usando à linguagem humana). OS LLM's requerem recursos computacionais significativos para processar dados, por isso utilizam unidades de processamento gráfico (GPUs), para acelerar o treinasmento e a opoeração dos chamados transformers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3447078-f291-4ded-8621-2165489ca052"/>
</p>

Source: Nvidia 

> Transformers são uma arquitetura que transformam ou alteram uma sequência de entra em uma sequência de saída de forma contínua, foi apresentada pelo paper "Attention is All you Need". Sua arquitetura melhora a capacidade dos modelos de Deep Learning ao capturar dependências contextuais em sequências de dados, como palavras em uma frase. As arquiteturas anteriores Recurrent Neural Network (RNN) e Long short-term memory (LSTM) perdiam detalhes em sequências longas, pois processavam a sequência passo a passo, ou seja, um elemento de cada vez. Os Transformers introduzem o mecanismo de Self-Attention, que olha todos os elementos da sequência de uma vez, ou seja, capturam as relações contextuais entre todas as partes de um sequência simultanemante (contexto). Por exemplo, no prompt de entrada "Qual é a cor do céu?", o modelo indentifica a relação entre "cor", "céu" e "azul" para gerar o prompt de saída "O céu é azul".

<p align="center">
  <img src="https://github.com/user-attachments/assets/ce0dd06f-c221-4ed8-ba49-72acc13f7bb5"/>
</p>

### Embeddings
> São representações numéricas de dados textuais, usados para transformar palavras ou sentenças em vetores númericos que o modelo possa entender e processar, permitindo que capturar o significado semâtico do texto. O modelo aprende a separar e agrupar esses extos com base em suas similaridades. Então por exemplo, quando o modelo recebe uma palavra nova, como "maça", ele sabe exatamente onde colocar, muito provavalmente em um bloco onde estão outras frutas.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c2d910e-d27c-45e8-a4f7-84af189887b8"/>
</p>
### Tokens
> São unidades menores de texto criadas através do processo de tokenização, que converter texto em números

### Prompt Engineering

- Testar Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- https://www.promptingguide.ai/introduction
- https://lmarena.ai/
- https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
- Gerador de Prompts https://www.feedough.com/ai-prompt-generator/

> A engenharia de prompt é a ciência empírica de planejar, criar e testar prompts para gerar melhores respostas em grandes modelos de linguagem.

### Hugging Face

### Langchain

### MCP

### RAG 
