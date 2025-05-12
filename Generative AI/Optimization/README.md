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

![chatgpt](https://github.com/user-attachments/assets/e2e077ff-08ad-4db2-9b7f-a7cc3acaa00d)

### Techniques
- colocar tabela 

### Tools
- Guide = https://www.promptingguide.ai/
- Test Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Generator of Prompts = https://www.feedough.com/ai-prompt-generator/
- Ranking of LLM's = https://beta.lmarena.ai/leaderboard

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

Criar seu próprio prompt pode ser o ideal caso deseje alcançar casos muito específicos. Mas caso esteja sem muito tempo para experimentar (ou não sabe muito bem o melhor modo) uma boa dica é procurar por prompts na internet.

Existem diversos sites e repositórios que disponibilizam prompts feitos pela comunidade.

Um exemplo é hub do LangSmith: https://smith.langchain.com/hub Ele faz parte do ecossistema LangChain. Isso será muito conveniente mais tarde, pois veremos como puxar prompts hospedados lá através de apenas uma função

![20250512_010859444_iOS](https://github.com/user-attachments/assets/bc1fa200-861c-43ba-a803-d2610a1bd0c6)

### Formatando Mensagem 

Um caso de uso cada vez mais comum para LLMs é o chat. Em um contexto de chat, em vez de continuar uma única sequência de texto (como é o caso com um modelo de linguagem padrão), o modelo continua uma conversa que consiste em uma ou mais mensagens, cada uma das quais inclui uma função, como "usuário" ou "assistente", bem como texto da mensagem.

Portanto, o prompt também pode ser estruturado dessa forma abaixo. Veremos com mais detalhes quando estivermos usando o LangChain, pois teremos recursos adicionais e que vão melhorar o uso deste modo

- msg: Esta lista contém a mensagem de entrada à qual queremos que o modelo responda. O formato da mensagem inclui um dicionário com as chaves role e content.
- role: "user" indica que a mensagem é do usuário. Outras funções possíveis podem incluir "system" ou "assistant" se você estiver simulando uma conversa com vários turnos. Diferentes modelos podem possuir roles com nomes diferentes, aqui com o Phi 3 são esperados esses.
- content: Aqui deixamos a pergunta real que queremos que o modelo responda, no caso, o nosso prompt.

### Quantização

técnicas de quantização e o BitsAndBytesConfig da biblioteca transformers, é possível carregar e executar modelos massivos de forma eficiente, sem comprometer significativamente o desempenho. As técnicas de quantização reduzem os custos de memória e computação ao representar pesos e ativações com tipos de dados de menor precisão, como inteiros de 8 bits (int8). Isso permite carregar modelos maiores e acelerar a inferência. usaremos o BitsAndBytesConfig para habilitar a quantização de 4 bits. Essa configuração ajuda a reduzir o footprint de memória e a carga computacional, tornando viável o uso de modelos grandes em recursos de hardware limitados. link: Mais sobre quantização aqui: https://huggingface.co/blog/4bit-transformers-bitsandbytes
Ainda existem outras soluções para quantização (por exemplo AutoGPTQ ou o AutoAWQ) que podem ou não otimizar mais ainda. Caso não deseje se incomodar com otimização/desempenho e ao mesmo tempo manter a qualidade então avalie a possibilidade de usar uma solução paga.

![20250512_012754624_iOS](https://github.com/user-attachments/assets/d63b0dd9-4b59-424e-8550-7ff24bd49e1e)

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


# RAG

# MCP

