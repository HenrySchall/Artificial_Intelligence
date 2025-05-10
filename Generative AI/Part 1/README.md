# Foundation Models

'# Uninstall Packages # pip freeze > packages.txt# pip uninstall -y -r packages.txt'
https://huggingface.co/docs/transformers/en/model_doc/cohere

### Ranking
https://beta.lmarena.ai/leaderboard


> Foundation Models são modelos de inteligência artificial (AI) de grande escala, baseados em técnicas de Deep Learning (Redes Neurais Profundas) de aprendizado não-supervisionado, ou seja, sem qualquer instrução humana. Eles são utilizados em enormes volumes de dados para criar conexões e captura as relações complexas entre os dados, para depois podem ser adaptados para tarefas específicas. O termo foundation foi popularizado pela Universidade de Stanford em 2021, devido ao fato que eles funcionam como base para construção de outras aplicações, em vez de treinarem um modelo do zero para cada tipo problema. Os Large Language Models (LLM's) são uma classe dos Foundation Models, especificos para linguagem, portando focam em tarefas relacionadas a texto, nesse contexto que surge os conceitos de Prompt Engineering (prática de criar instruções (prompts) inteligentes para controlar e guiar o comportamento de modelos de linguagem) e Natural Language Processing (NLP) (prática de ensinar computadores a entender, interpretar, gerar e interagir usando à linguagem humana). OS LLM's requerem recursos computacionais significativos para processar dados, por isso utilizam unidades de processamento gráfico (GPUs), para acelerar o treinasmento e a opoeração dos chamados transformers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3447078-f291-4ded-8621-2165489ca052"/>
</p>

Source: Nvidia

### Transformers
Transformers são uma arquitetura que transformam ou alteram uma sequência de entra em uma sequência de saída de forma contínua, foi apresentada pelo paper "Attention is All you Need". Sua arquitetura melhora a capacidade dos modelos de Deep Learning ao capturar dependências contextuais em sequências de dados, como palavras em uma frase. As arquiteturas anteriores Recurrent Neural Network (RNN) e Long short-term memory (LSTM) perdiam detalhes em sequências longas, pois processavam a sequência passo a passo, ou seja, um elemento de cada vez. Os Transformers introduzem o mecanismo de Self-Attention, que olha todos os elementos da sequência de uma vez, ou seja, capturam as relações contextuais entre todas as partes de um sequência simultanemante (contexto). Por exemplo, no prompt de entrada "Qual é a cor do céu?", o modelo indentifica a relação entre "cor", "céu" e "azul" para gerar o prompt de saída "O céu é azul".

<p align="center">
  <img src="https://github.com/user-attachments/assets/72f25195-e576-4d6c-8148-e6f236ac2190"/>
</p>

source:brains.dev

### Embeddings & Tokens
Os Embeddings são representações vetoriais numéricas de dados textuais, usados para transformar palavras ou sentenças em vetores númericos que o modelo possa entender e processar, permitindo capturar o significado semâtico do texto. O modelo aprende a separar e agrupar esses extos com base em suas similaridades. Então por exemplo, quando o modelo recebe uma palavra nova, como "maça", ele sabe exatamente onde colocar, muito provavalmente em um bloco onde estão outras frutas. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/bfea28f0-0a3a-449a-923e-2f0655e3d766"/>
</p>

> Já os tokens são unidades menores de texto criadas através do processo de tokenização, que converter texto em números. Sendo assim o Embedding é a representação vetorial do texto em um espaço multidimensional, o Token é uma representação estática de unidades de texto.

<p align="center">
  <img src="https://github.com/user-attachments/assets/efba8ea9-338a-4ff0-89e8-f8e90cb5a213"/>
</p>
https://platform.openai.com/tokenizer

### Modelos
proprietarios (closed source) ou de códgiosd aberttos (open Source)
SLM -> Small Languagew Models = feito em ambientes locais

### Configuração Nvidia Cuda
https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html#f1

Install Cuda Toolkit 

Install cuDNN Toolkit  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 (https://pytorch.org/get-started/locally/)

Observe que a versão do Cuda Toolkit deve ser compátivel com o cuDNN Toolkit e com a versão do Framework escolhido (Pytorch, Tensorflow ou Keras)

Example: Pytorch Cuda 11.8 -> cuDNN 9.8.0 for CUDA 11.x


## Hugging Face

Tipos de LLM's 

base models -. passam apenas pelo pré-trainamento e completam textos com palavras prova´veis (bons para prever palavras subsequentes entao se geramos um pergunta ele retorna com uma perguibnta (projetado para completaer frases prevendo as palavras mais provaveis com base no texto anterior

Modelos instruct-tuned -> modelos ajustados para instrulçoes passam por umaetapa adicional de ajuste parea instruções melhorand a capacidade de seguir comandos especificos (ESPERAM ser solicitados a fazer algo

Modelos de Chat -> foram ajustados para dfuncionar em chatbots, poretanto podem sewermais apropriados para conversas.
Esperando o envolvimentos em uma conversa

modelos nao ajusrados para instruções simplesmente geram uma saide que continua a parti do prompt

fine-tuning -. treinar um parte do modelo especifica para um especfiico cenario (base de dados menor e mais especializada)

## Langchain

> É uma biblioteca de código aberto projetada para facilitar a integração de LLM's como

> Modelos de LLM's disponíveis https://python.langchain.com/v0.2/docs/integrations/llms/

> Devido ao fato do LangChain se integrar bem ao Hugging Face, podemos usar a mesma lógica de pipeline que vimos antes na implementação com o Transformers. São uma ótima e fácil maneira de usar modelos para inferência.quantização e também a tokanização.












Componentes
- Modelos
- Prompts
- Chains = encaderar LLM's m aplicações complexas, permitindo a ligação entre múltiplos modelos ou outros módulos especializados
- Memória: Módulos que permitem o gerenciamento e alteração de conversas anteriores, essencial para chatbots que preciam relembrar interações passadas para manter coerência
- agentes = euqipads com kit de ferramentas abrangentes, que possibvvilita escolher quais ferramentas usar com base nas informações do usuário
- Indices = metod spara organizar documentos 

Econosistema
- langchain community - integraç~~oes com terceiros tipo langchain-openai
- lanchiga chain = chains, agentes e estrategias de retrivel quwe compoem a arquiteutra cognitiva de uma aplicação 
- langraph: para construir aplicações robustas e como estado para múltiplos atores com LLM's, modelando etapas como arestas e nós em um gráfico. Integra-se perfeitamente com Langchain, mas pode ser usado sem ele
- langserve: Para implementar chains do lang cghain com oapis rest 
- langsmith platadfoma para desenvolver de aplicacoes LLM

## MCP
## RAG


# Prompt Engineering
> Prompt Engineering is the practice of designing, testing, and refining prompts to better understand the capabilities and limitations of Large Language Models (LLMs). Researchers use prompt engineering to improve the capabilities of LLMs across a wide range of tasks, and developers use prompt engineering to design robust and effective prompting techniques to interact with LLMs and other tools. Prompt Engineering is the development of more accurate, relevant, and controllable prompts.

### Avoiding hallucinations
- **Humanize**: Allow the model to say it doesn't know the answer and set it up for you to challenge it, letting it take a constructive approach as if it were a thought partner.
- **Include Context**: Find theoretical evidence (relevant excerpts) in the given context that support your answer (use of `<Quotes>`). Example:

`<Quotes>`

`The Industrial Revolution began in England in the 18th century...`

`</Quotes>`

`Based on the context above, answer:`

`1. What was the main cause of the Industrial Revolution?`

`2. Quote the exact passage that justifies your answer, within the <Quotes>…</Quotes> tag.`

- **Model Temperature:** Adjust the temperature of the model to control creativity, i.e. use low values ​​for more factual responses.
- **Self-Consistency:** Generate multiple responses and ask the model to develop a consensus.
- **Delimiters**: Use delimiters such as `--` or XML tags (`<tag>content</tag>`) to separate sections of the prompt, such as instructions, examples, and data. This helps the model distinguish different parts of the prompt.

### Basic Prompt Structure

![chatgpt prompt  frameworks  chatgptricks_page-0001](https://github.com/user-attachments/assets/e2e077ff-08ad-4db2-9b7f-a7cc3acaa00d)

### Techniques

### Tools
- Guide = https://www.promptingguide.ai/
- Test Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Generator of Prompts https://www.feedough.com/ai-prompt-generator/

## Repository Bibliographic References:
- Large Language Models are Zero-short Reasoners Contrastive Chin of Thought prompting, by Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa. 
- Guiding Large Language Models via Directional Stimulus Prompting, by Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan. 
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, by Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan. 
- Self-Consistency Improves Chain of Thought Reasoning in Language Models, by Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. 
- Deliberate Problem Solving with Large Language Models, by Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan.
- Generative Agents: Interactive Simulacra of Human Behavior, by Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein.
- Large Language Model Guided Tree-of-Thought, by Jieyi Long.
- Skeleton of Thought: Augmenting Language Models with Variable-Depth Reasoning Ability, by Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, Yu Wang.
- Generated Knowledge Prompting for Commonsense Reasoning, by Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, Hannaneh Hajishirzi
- Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations, by Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, Yejin Choi
- Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks, by Katja Filippova. 
- PAL: Program-aided Language Models. by Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig.
- Synergizing Reasoning and Acting in Language Models, by Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- Meta Prompting for AI Systems, by Yifan Zhang, Yang Yuan, Andrew Chi-Chih Yao.
- ART: Automatic multi-step reasoning and tool-use for large language models, by Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, Marco Tulio Ribeiro.
- Large Language Models Are Human-Level Prompt Engineers, by Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba.
- Active Prompting with Chain-of-Thought for Large Language Models, by Shizhe Diao, Pengcheng Wang, Yong Lin, Rui Pan, Xiang Liu, Tong Zhang.
- Reflexion: Language Agents with Verbal Reinforcement Learning, by Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao.
- Multimodal Chain-of-Thought Reasoning in Language Models, by Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, Alex Smola
- GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks, by Zemin Liu, Xingtong Yu, Yuan Fang, Xinming Zhang
