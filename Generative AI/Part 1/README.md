# Foundation Models

> Foundation Models são modelos de inteligência artificial (AI) de grande escala, baseados em técnicas de Deep Learning (Redes Neurais Profundas) de aprendizado não-supervisionado, ou seja, sem qualquer instrução humana. Eles são utilizados em enormes volumes de dados para criar conexões e captura as relações complexas entre os dados, para depois podem ser adaptados para tarefas específicas. O termo foundation foi popularizado pela Universidade de Stanford em 2021, devido ao fato que eles funcionam como base para construção de outras aplicações, em vez de treinarem um modelo do zero para cada tipo problema. Os Large Language Models (LLM's) são uma classe dos Foundation Models, especificos para linguagem, portando focam em tarefas relacionadas a texto, nesse contexto que surge os conceitos de Prompt Engineering (prática de criar instruções (prompts) inteligentes para controlar e guiar o comportamento de modelos de linguagem) e Natural Language Processing (NLP) (prática de ensinar computadores a entender, interpretar, gerar e interagir usando à linguagem humana). OS LLM's requerem recursos computacionais significativos para processar dados, por isso utilizam unidades de processamento gráfico (GPUs), para acelerar o treinasmento e a opoeração dos chamados transformers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3447078-f291-4ded-8621-2165489ca052"/>
</p>

Source: Nvidia

#### Transformers
Transformers são uma arquitetura que transformam ou alteram uma sequência de entra em uma sequência de saída de forma contínua, foi apresentada pelo paper "Attention is All you Need". Sua arquitetura melhora a capacidade dos modelos de Deep Learning ao capturar dependências contextuais em sequências de dados, como palavras em uma frase. As arquiteturas anteriores Recurrent Neural Network (RNN) e Long short-term memory (LSTM) perdiam detalhes em sequências longas, pois processavam a sequência passo a passo, ou seja, um elemento de cada vez. Os Transformers introduzem o mecanismo de Self-Attention, que olha todos os elementos da sequência de uma vez, ou seja, capturam as relações contextuais entre todas as partes de um sequência simultanemante (contexto). Por exemplo, no prompt de entrada "Qual é a cor do céu?", o modelo indentifica a relação entre "cor", "céu" e "azul" para gerar o prompt de saída "O céu é azul".

<p align="center">
  <img src="https://github.com/user-attachments/assets/72f25195-e576-4d6c-8148-e6f236ac2190"/>
</p>

source:brains.dev

#### Embeddings & Tokens
Os Embeddings são representações vetoriais numéricas de dados textuais, usados para transformar palavras ou sentenças em vetores númericos que o modelo possa entender e processar, permitindo capturar o significado semâtico do texto. O modelo aprende a separar e agrupar esses extos com base em suas similaridades. Então por exemplo, quando o modelo recebe uma palavra nova, como "maça", ele sabe exatamente onde colocar, muito provavalmente em um bloco onde estão outras frutas. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/bfea28f0-0a3a-449a-923e-2f0655e3d766"/>
</p>

> Já os tokens são unidades menores de texto criadas através do processo de tokenização, que converter texto em números. Sendo assim o Embedding é a representação vetorial do texto em um espaço multidimensional, o Token é uma representação estática de unidades de texto.

<p align="center">
  <img src="https://github.com/user-attachments/assets/efba8ea9-338a-4ff0-89e8-f8e90cb5a213"/>
</p>
https://platform.openai.com/tokenizer


#### Tipos de LLM's 

- Base models -. passam apenas pelo pré-trainamento e completam textos com palavras prova´veis (bons para prever palavras subsequentes entao se geramos um pergunta ele retorna com uma perguibnta (projetado para completaer frases prevendo as palavras mais provaveis com base no texto anterior palavra subsequentes - nao gera resposta g

- Instruct-Tuned -> modelos ajustados para instrulçoes passam por umaetapa adicional de ajuste parea instruções melhorand a capacidade de seguir comandos especificos (ESPERAM ser solicitados a fazer algo

- Chat -> foram ajustados para funcionar em chatbots, poretanto podem ser mais apropriados para conversas.
Esperando o envolvimentos em uma conversa

instruct -> seguir as instruções fornecidas

CHat -< esperam estar envolvidos em um conversa com diferentes atores

modelos nao ajustados para instruções simplesmente geram uma saide que continua a parti do prompt

fine-tuning -Tecnica treinar um parte do modelo especifica para um especfiico cenario (base de dados menor e mais especializada)
transformar de base model para intruct-tuned, Envolve pegar o mnodelo base pre-treinado e treina-lo mais com um datasert menor e mauis especializado relevatge para a tarefa desejada


LLM's sao projetados para completar frases prevendo as palavras mais provaveis com base n otexto anterior entao os modelos bases funcionam dessa forma, pdoeriamo dar algum dica para ele tipo um pergunta onde os poinguins viver e ja dar uma parte da resposta, isso e a egnhearia de prompt (escolhare das melhores plavras para a AI) maximizara resposta



## Hugging Face

> O Hugging Face é uma empresa que começou na França em 2017, inicialmente focada no desenvolvimento de chatbots. Com o tempo, a empresa evoluiu para criar uma infraestrutura própria para o processamento de linguagem natural (NLP), oferecendo uma série de bibliotecas de Python que simplificam o uso de modelos de NLP. Hoje, o Hugging Face é um hub de modelos open-source, contendo diversos modelos pré-treinados que podem ser utilizados para desenvolvimento de soluções de Gen AI.

```
id_model =  "microsoft/Phi-3-mini-4k-instruct"
```

- microsoft = Organização desenvolvedora.
- Phi-3-mini = Família do modelo. 
- 4k = Tamanho da Janela de Contexto ou Sequência de Tokens (4000 mil tokens em uma única entrada, para gerar uma sequência de texto).
- instruct = Tipo de Modelo.

```
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")
```

- device_map = Especifica que o modelo deve ser processado usando GPU.
- torch_dtype = Define o tipo de dados apropriado para os tensores do modelo.
- trust_remote_code = Permite o carregamento de código personalizado do repositório de modelos no HuggingFace.
- attn_implementation = Especifica o método de implementação para o mecanismo de Self-Attention. Sendo a configuração "eager" uma implementação particular.

#### Tokenizador
> Preparar os dados para realizar o processo de converção de texto bruto em tokens (representações númericas), ou seja, permite o modelo interpretar o texto digitado.

#### Pipeline 
> É uma abstração que simplifica o uso de modelos pré-treinados para uma variedade de tarefas, facilitando o processo de execução de processamento de linguagem natural, devido a sua interface unificada.
```
tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model , tokenizer = tokenizer) # Criação de pipeline

```

- "text-generation" = Especifica a tarefa que o pipeline está configurado para executar.
- model = Especifica o modelo que o pipeline usará.
- tokenizer = Especifica o tokenizador que o pipeline usará. 

```
# Especificando paramâtros do modelo
generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.1, "do_sample": True}

```

- "max_new_tokens" = Comprimento do texto gerado 
- "return_full_text = Se deve fornecer o texto completo (Prompt de entrada + resposta)
- "temperature" = Controle de aleatoriedade do processo ( 0.1 - determinística < 0.5 > criativa - 0.9)
- "do_sample" = Define se a amostragem aleatória será usada na geração das próximas palavras do texto.
  
  - Quando do_sample = True, as próximas palavras são geradas com base na distribuição de probabilidades, permitindo variação e criatividade (o texto pode mudar, com o mesmo prompt).
  - Quando o do_sample = False, escolhe-se sempre a palavra mais provável (argmax). Isso gera respostas mais previsíveis e determinísticas.
 
> Repare que o modelo continuou gerando depois de dar a resposta, até por isso dessa vez demorou mais. O que acontece é que o modelo continua "conversando sozinho", como se simulasse uma conversa. É um comportamento esperado já que não definimos o que chamamos de token de parada (end token). Isso será explicado com detalhes, mas por enquanto o que você precisa saber é que para evitar esse comportamento nós utilizamos templates, que são recomendados pelos próprios autores geralmente (ou pela comunidade). Uma forma de consertar isso são os templates
 
#### Templates 

> Os modelos (templates) de prompt ajudam a traduzir a entrada e os parâmetros do usuário em instruções para um modelo de linguagem. Isso pode ser usado para orientar a resposta de um modelo, ajudando-o a entender o contexto e gerar saída relevante e mais coerente. <|##nome##|> -> Tokens especiais (special tokens) usados para delimitar o início e fim de um texto e dizer ao modelo como queremos que a mensagem seja interpretada. Tipos:

- <|system|>, <|user|> e <|assistant|>: correspondem ao papel (role) das mensagens. Os papéis usados aqui são: system, user e assistant
- <|end|>: Isso é equivalente ao token EOS (End of String), usado para marcar o fim do texto/string.



# Usaremos o .format para concatenar o prompt nesse template, assim não precisamos redigitar ali manualmente



