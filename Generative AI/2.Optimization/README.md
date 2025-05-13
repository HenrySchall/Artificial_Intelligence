# Prompt Engineering
> Prompt Engineering é a ciência focada em desenvolver, testar e otimizar as entradas (prompts) para modelos de inteligência artificial generativa, como os Large Language Models (LLM's), buscando obter resultados de forma mais eficaz e precisa. Tenha em mente que, as LLM's são projetadas para completar frases, ou seja, elas sempre vão trabalhar prevendo as palavras mais prováveis com base no texto anterior, sendo assim dependendo as informações fornecidas, as saídas (resultados) serão diferentes.

### Tools
- Guide = https://www.promptingguide.ai/
- Test Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Generator of Prompts = https://www.feedough.com/ai-prompt-generator/
- Ranking of LLM's = https://beta.lmarena.ai/leaderboard

### Softwares de Desenvolvimento
- Gale, by Kore.AI
- Vellum
- Zapier
- N8N
- Baseplate
- LangSmith
- Everyprompt
- Flux
- Dyno
- ChainForge
- Prompt Stack
- Prisms
- PromptTools
- Spellbook
- Promptify

### Basic Prompt Structure

![chatgpt](https://github.com/user-attachments/assets/e2e077ff-08ad-4db2-9b7f-a7cc3acaa00d)

### Dicas para evitar alucinações
- **Humanize**: Permita que o modelo diga não saber a resposta e também deixe ele te questionar, desenvolva uma abordagem construtiva como se fosse um parceiro de pensamento.
- **Contexto**: Encontre evidências teóricas (trechos relevantes) no contexto fornecido que apoiem sua resposta (use `<Citações>`). Exemplo:
- **Temperatura**: Ajuste a temperatura do modelo para controlar a criatividade
- **Autoconsistência**: Gere múltiplas respostas e peça ao modelo para chegar a um consenso.
- **Delimitadores**: Use delimitadores como `--` ou tags (`<tag>content</tag>`) para separar seções do prompt, como instruções, exemplos e dados. Isso ajuda o modelo a distinguir diferentes partes do prompt.


### Techniques
- colocar tabela 

### Prompt Engineering com HuggingFace 

```
# Example 1
prompt = "Gere um código em python que escreva a sequência de fibonnaci"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

# Example 2
prompt = "Gere um código em python que escreva a sequência de fibonnaci"

sys_prompt = "Você é um programador experiente. Retorne o código requisitado e forneça explicações breves se achar conveniente" - Adiciona o modo de comportamento da AI

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])
```

### Formatando Saídas

> O uso cada vez mais comum para LLMs é o chat. Em um contexto de chat, em vez de continuar uma única sequência de texto (como é o caso com um modelo de linguagem padrão), o modelo continua uma conversa que consiste em uma ou mais mensagens, cada uma das quais inclui uma função, como "usuário" ou "assistente", bem como texto da mensagem.

- Messages: Representação estruturada da mensagem
- Role: Subdivido em 3 parâmtros:
    - User = Entrada/comando do usuário
    - Assistant = Parâmetro opcional, utilizado para modelos com suporte ao ChatML (Conversas Contínuas)
    - System = Parâmetro opcional, fornecer instruções iniciais, contexto ou configurações de comportamento geral do modelo
- Content: Aqui deixamos a pergunta real que queremos que o modelo responda, no caso, o nosso prompt.

```
# Example 1
prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

messages = [
    {"role": "system", "content": prompt_sys},
    {"role": "user", "content": prompt}
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

# Example 2 
id_model = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
generation_args = {"max_new_tokens": 100, "do_sample": True, "temperature": 0.7, "return_full_text": False}

prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": prompt_sys},
    {"role": "assistant", "content": "Claro! Aqui estão: Paris, Roma, Londres, Berlim, Madri, Viena, Amsterdã, Lisboa, Praga e Barcelona."},
    {"role": "user", "content": "Qual delas é a melhor para visitar em dezembro?"},
    {"role": "assistant", "content": ""}  # Modelo completará a próxima resposta
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

## Templates 

> Os modelos (templates) de prompt ajudam a traduzir a entrada e os parâmetros do usuário em instruções para um modelo de linguagem. Isso pode ser usado para orientar a resposta de um modelo, ajudando-o a entender o contexto e gerar saída relevante e mais coerente. <|##nome##|> -> Tokens especiais (special tokens) usados para delimitar o início e fim de um texto e dizer ao modelo como queremos que a mensagem seja interpretada. Tipos:

https://huggingface.co/docs/transformers/chat_templating

- <|system|>, <|user|> e <|assistant|>: correspondem ao papel (role) das mensagens. Os papéis usados aqui são: system, user e assistant
- <|end|>: Equivalente ao token EOS (End of String), usado para marcar o fim do texto/string.

```
prompt = "Explique o que é computação quântica"

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt) # .format para concatenar o prompt nesse template, assim não precisamos redigitar ali manualment

output = pipe(template, **generation_args)
print(output[0]['generated_text'])
```

```

```

Os tokens especiais usados para interagir via prompt com o Llama 3 são esses:

    <|begin_of_text|>: equivalente ao token BOS (Beginning of String), indicando o início de uma nova sequência de texto.

    <|eot_id|>: indica o fim de uma mensagem.

    <|start_header_id|>{role}<|end_header_id|>: esses tokens envolvem o papel de uma mensagem específica. Os papéis possíveis são: system, user e assistant.

    <|end_of_text|>: Isso é equivalente ao token EOS (End of String). Ao chegar nesse token, o Llama 3 deixará de gerar mais tokens.

https://huggingface.co/collections/eduagarcia/portuguese-llm-leaderboard-best-models-65c152c13ab3c67bc4f203a6

Alguns modelos de linguagem pegam uma lista de mensagens como entrada e retornam uma mensagem. Existem alguns tipos diferentes de mensagens. Todas as mensagens têm uma propriedade role, content e response_metadata.

A função (role) descreve quem está dizendo a mensagem (ex: human, system). LangChain tem diferentes classes de mensagem para diferentes funções

A propriedade conteúdo (content) descreve o conteúdo da mensagem, podendo ser:

    Uma string (a maioria dos modelos lida com esse tipo de conteúdo)
    Uma lista de dicionários (isso é usado para entrada multimodal, onde o dicionário contém informações sobre esse tipo de entrada e esse local de entrada)

A propriedade response_metadata contém metadados adicionais sobre a resposta. Os dados aqui são frequentemente específicos para cada provedor de modelo. É aqui que informações como log-probs (probabilidades de log) e uso de token podem ser armazenadas. Certos modelos de chat podem ser configurados para retornar probabilidades de log em nível de token, representando a probabilidade de um determinado token. Por exemplo, para isso pode-se usar essa sintaxe: msg.response_metadata["logprobs"]["content"][:5]

https://python.langchain.com/v0.2/docs/concepts/#messages

Os modelos de prompt (Prompt Templates) ajudam a traduzir a entrada e os parâmetros do usuário em instruções para um modelo de linguagem. Pode ser usado para orientar a resposta de um modelo, ajudando-o a entender o contexto e gerar uma saída relevante e coerente baseada em linguagem. Isso principalmente facilita a criação de prompts de maneiras variáveis. Com o Langchain, temos uma maneira eficiente de conectar isso aos diferentes LLMs que existe. Para mudar a LLM, basta alterar o código anterior de carregamento, e o código seguinte permanece igual. Ou seja, é um modo muito mais eficiente caso esteja querendo desenvolver aplicações profissionais.

https://python.langchain.com/v0.2/docs/concepts/#prompt-templates

Existem alguns tipos diferentes de modelos de prompt:


# RAG

# MCP

