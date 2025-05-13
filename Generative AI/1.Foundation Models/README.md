# Foundation Models

> Foundation Models são modelos de inteligência artificial (AI) de grande escala, baseados em técnicas de Deep Learning (Redes Neurais Profundas) de aprendizado não-supervisionado, ou seja, sem qualquer instrução humana. Eles são utilizados em enormes volumes de dados para criar conexões e captura as relações complexas entre os dados, para depois podem ser adaptados para tarefas específicas. O termo foundation foi popularizado pela Universidade de Stanford em 2021, devido ao fato que eles funcionam como base para construção de outras aplicações, em vez de treinarem um modelo do zero para cada tipo problema. Os Large Language Models (LLM's) são uma classe dos Foundation Models, especificos para linguagem, portando focam em tarefas relacionadas a texto, nesse contexto que surge os conceitos de Prompt Engineering (prática de criar instruções (prompts) inteligentes para controlar e guiar o comportamento de modelos de linguagem) e Natural Language Processing (NLP) (prática de ensinar computadores a entender, interpretar, gerar e interagir usando à linguagem humana). OS LLM's requerem recursos computacionais significativos para processar dados, por isso utilizam unidades de processamento gráfico (GPUs), para acelerar o treinasmento e a opoeração dos chamados transformers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3447078-f291-4ded-8621-2165489ca052"/>
</p>

Source: Nvidia

### Transformers
> Transformers são uma arquitetura que transformam ou alteram uma sequência de entra em uma sequência de saída de forma contínua, foi apresentada pelo paper "Attention is All you Need". Sua arquitetura melhora a capacidade dos modelos de Deep Learning ao capturar dependências contextuais em sequências de dados, como palavras em uma frase. As arquiteturas anteriores Recurrent Neural Network (RNN) e Long short-term memory (LSTM) perdiam detalhes em sequências longas, pois processavam a sequência passo a passo, ou seja, um elemento de cada vez. Os Transformers introduzem o mecanismo de Self-Attention, que olha todos os elementos da sequência de uma vez, ou seja, capturam as relações contextuais entre todas as partes de um sequência simultanemante (contexto). Por exemplo, no prompt de entrada "Qual é a cor do céu?", o modelo indentifica a relação entre "cor", "céu" e "azul" para gerar o prompt de saída "O céu é azul".

<p align="center">
  <img src="https://github.com/user-attachments/assets/72f25195-e576-4d6c-8148-e6f236ac2190"/>
</p>

source:brains.dev

### Embeddings & Tokens
> Embeddings são representações vetoriais numéricas de dados textuais, usados para transformar palavras ou sentenças em vetores númericos que o modelo possa entender e processar, permitindo capturar o significado semâtico do texto. O modelo aprende a separar e agrupar esses extos com base em suas similaridades. Então por exemplo, quando o modelo recebe uma palavra nova, como "maça", ele sabe exatamente onde colocar, muito provavalmente em um bloco onde estão outras frutas. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/bfea28f0-0a3a-449a-923e-2f0655e3d766"/>
</p>

> Já os tokens são unidades menores de texto criadas através do processo de tokenização, que converter texto em números. Sendo assim o Embedding é a representação vetorial do texto em um espaço multidimensional, o Token é uma representação estática de unidades de texto.

<p align="center">
  <img src="https://github.com/user-attachments/assets/efba8ea9-338a-4ff0-89e8-f8e90cb5a213"/>
</p>
https://platform.openai.com/tokenizer

### Tipos de LLM's 

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

### Tokenizador
> Preparar os dados para realizar o processo de converção de texto bruto em tokens (representações númericas), ou seja, permite o modelo interpretar o texto digitado.

### Pipeline 
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

## Langchain

> 

### Componentes 

- Modelos: Oferece uma interface padrão para interações com uma ampla gama de LLMs.

- Prompts: Ferramentas para simplificar a criação e tratamento de prompts dinâmicos.

- Chains (Corrente, Cadeia ou Sequencia): Interface padrão para encadear LLMs em aplicações complexas, permitindo a ligação entre múltiplos modelos ou outros módulos especializados.

- Memória: Módulos que permitem o gerenciamento e alteração de conversas anteriores, essencial para chatbots que precisam relembrar interações passadas para manter coerência.

- Agentes: Equipados com um kit de ferramentas abrangente, podem escolher quais ferramentas usar com base nas informações do usuário.

- Índices: Métodos para organizar documentos (que contém dados proprietários, por exemplo) de forma a facilitar a interação eficaz com LLMs.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cf01f65a-81b7-4f79-b6ef-dbe5c82af5cb"/>
</p>

### Ecossistema 

• langchain-core: Abstrações básicas e LangChain Expression Language (LCEL).

• langchain-community: Integrações de terceiros. Pacotes parceiros (por exemplo, langchain-openai, langchain-anthropic, etc.): Algumas integrações foram divididas em seus próprios pacotes leves que dependem apenas do langchain-core.

• langchain: Chains, Agentes e Estratégias de Retrieval que compõem a arquitetura cognitiva de uma aplicação.

• LangGraph: Para construir aplicações robustas e com estado para múltiplos atores com LLMs, modelando etapas como arestas e nós em um gráfico. Integra-se perfeitamente com LangChain, mas pode ser usado sem ele.

• LangServe: Para implementar chains do LangChain como APIs REST.

• LangSmith: Uma plataforma para desenvolvedores que permite depurar, testar, avaliar e monitorar aplicações LLM's

![userlm](https://github.com/user-attachments/assets/0e3d7455-9f4a-4a3c-93cb-ae795cd38331)

> A integração de Hugging Face com LangChain traz diversos benefícios,
conforme citado anteriormente.• Hugging Face disponibiliza uma ampla variedade de modelos pré-treinados
que podem ser facilmente incorporados às suas aplicações. • LangChain facilita essa incorporação, fornecendo uma interface uniforme eferramentas extras para melhorar desempenho e eficiência. • Com essas ferramentas combinadas, você pode se focar mais na lógica de negócios e menos nas questões técnicas

```
model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model = model, tokenizer = tokenizer, task = "text-generation", temperature = 0.1, max_new_tokens = 500, do_sample = True, repetition_penalty = 1.1, return_full_text = False)
llm = HuggingFacePipeline(pipeline = pipe)
```
- do_sample = Indica que a geração deve envolver amostragem aleatória (em vez de escolher sempre o token mais prováve
- repetition_penalty = Penaliza repetições de tokens já gerados. Valores acima de 1 reduzem a chance de repetições
- return_full_text = Se False, a saída retorna apenas o texto gerado, sem incluir o prompt original.

### Quantização

> A quantização reduz a precisão dos números usados para representar os parâmetros de um modelo, diminuindo o footprint (uso) de memória e carga computacional, possibilitando carregar e executar modelos massivos de forma eficiente, sem comprometer significativamente o desempenho. O processo consiste em usar números de ponto flutuante de 16 bits (float16) ou de 8 bits (int8), ao invés de 32 bits (float32). Dentre os Frameworks disponíveis para realizar o processos existem o  BitsAndBytesConfig, AutoGPTQ e AutoAWQ, a escolha deles vai de preferência do usuário ou da performace do modelo sendo utilizado.
> Link: https://huggingface.co/blog/4bit-transformers-bitsandbytes

```
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)
```

- load_in_4bit- Este parâmetro habilita a quantização de 4 bits. Quando definido como True, os pesos do modelo são carregados com precisão de 4 bits, reduzindo significativamente o uso de memória.Impacto: Menor uso de memória e cálculos mais rápidos com impacto mínimo na precisão do modelo.
- bnb_4bit_quant_type - especifica o tipo de quantização de 4 bits a ser usado. "nf4" significa NormalFloat4, um esquema de quantização que ajuda a manter o desempenho do modelo enquanto reduz a precisão. Impacto: Equilibra o trade-off entre tamanho e desempenho do modelo.
- bnb_4bit_use_double_quant - quando definido como True, este parâmetro habilita a quantização dupla, o que reduz ainda mais o erro de quantização e melhora a estabilidade do modelo. Impacto: Reduz o erro de quantização, aprimorando a estabilidade do modelo.
- bnb_4bit_compute_dtype - define o tipo de dados para cálculos. Usar torch.bfloat16 (Brain Floating Point) ajuda a melhorar a eficiência computacional, mantendo a maior parte da precisão dos números de ponto flutuante de 32 bits. Impacto: Cálculos eficientes com perda mínima de precisão.

```
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

```
prompt = ("Quem foi a primeira pessoa no espaço?")
messages = [{"role": "user", "content": prompt}]
```

Recomendamos usar a função Hugging Face tokenizer.apply_chat_template(), que aplica automaticamente o modelo de chat correto para o respectivo modelo. É mais fácil do que escrever manualmente o modelo de chat e menos propenso a erros. return_tensors="pt" especifica que os tensores retornados devem ser no formato PyTorch.

As demais linhas de código: tokenizam as mensagens de entrada, movem os tensores para o dispositivo correto, geram novos tokens com base nos inputs fornecidos, decodificam os tokens gerados de volta em texto legível e finalmente retornam o texto gerado.

    model_inputs = encodeds.to(device) - Move os tensores codificados para o dispositivo especificado (CPU ou GPU) para serem processados pelo modelo.

    encodeds - Os tensores gerados na linha anterior. to(device) - Move os tensores para o dispositivo especificado (device), que pode ser uma CPU ou GPU.

    generated_ids = model.generate... -> Gera uma sequência de tokens a partir dos model_inputs.
        model.generate: Função do modelo que gera texto baseado nos inputs fornecidos.
        model_inputs: Os inputs processados, prontos para serem usados pelo modelo.
        max_new_tokens=1000: Limita a geração a no máximo 1000 novos tokens.
        do_sample=True: Habilita amostragem aleatória durante a geração, o que pode resultar em saídas mais variadas.
        pad_token_id=tokenizer.eos_token_id: Define o token de padding para ser o token de fim de sequência, garantindo que a geração seja corretamente terminada.

    decoded = tokenizer.batch_decode(generated_ids) - decodifica os IDs gerados de volta para texto legível.
        tokenizer.batch_decode - função que converte uma lista de IDs de tokens de volta para texto.
        generated_ids - os IDs dos tokens gerados na etapa anterior.

- res = decoded[0] - extrai o primeiro item da lista de textos decodificados. decoded[0]: Pega o primeiro texto da lista decoded, que corresponde à geração de texto para o primeiro (e possivelmente único) input fornecido.

```
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
generated_ids = model.generate(model_inputs, max_new_tokens = 1000, do_sample = True,
                               pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
res = decoded[0]
res
```


Você verá que, com o LangChain, teremos mais opções e ferramentas, pois a biblioteca oferece um ecossistema completo e integrado às principais e mais modernas soluções de modelos de linguagem, tanto abertas quanto privadas.

Então, por que pode ser interessante saber esse método que mostramos agora, se o LangChain é melhor e oferece mais opções? Pode ser útil caso você esteja testando um modelo novo e recém-publicado que ainda não possui tanta compatibilidade.

Mesmo com o LangChain, ao lidar com literalmente milhares de modelos diferentes, pode haver certa incompatibilidade ao carregá-los. Isso geralmente é corrigido pela equipe de desenvolvimento em algum release futuro, mas nem sempre é imediato - e outras soluções você encontrará apenas procurando em fóruns já que são publicados pela comunidade.

Portanto, saber esse método pode ser útil se você estiver testando os modelos open-source mais recentes que não carregaram corretamente com o LangChain.

Pode ser um pequeno inconveniente para alguns, mas é necessário entender que esse é o "preço" a se pagar por estar na fronteira e usar os modelos Open Source mais modernos e poder utilizá-los de forma gratuita.




