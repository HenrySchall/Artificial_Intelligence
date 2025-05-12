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

### Prompt Engineering com LangChain


# RAG

# MCP

