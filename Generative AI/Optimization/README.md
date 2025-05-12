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


# RAG

# MCP

