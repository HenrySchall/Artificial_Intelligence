# Prompt Engineering

> Prompt Engineering é a prática de projetar, testar e refinar entradas (prompts) para compreender melhor as capacidades e limitações dos Large Language Models (LLM's). Pesquisadores utilizam a engenharia de prompt para aprimorar a capacidade de LLM's em uma ampla gama de tarefas e desenvolvedores utilizam a engenharia de prompts para projetar técnicas de prompts robustas e eficazes para interagir com LLMs e outras ferramentas. Sendo assim, podemos dizer que Prompt Engineering é o desenvolvimento de prompts mais precisos, relevantes e controláveis.

### Evitando alucinações
- **Humanize**: Permita que o modelo diga que não sabe a resposta e configure ele para te contestar, deixe ele adotar uma abordagem construtiva como se fosse um parceiro de ideias.
- **Coloque Contexto**: Encontrar evidências teóricas (trechos relevantes) no contexto fornecido que embasem sua resposta (uso de `<Quotes>`). Exemplo:

`<Quotes>`

`A Revolução Industrial começou na Inglaterra no século XVIII...`

`</Quotes>`

`Com base no contexto acima, responda:`

`1. Qual foi a principal causa da Revolução Industrial?`

`2. Cite o trecho exato que justifica sua resposta, dentro da tag <Quotes>…</Quotes>`

- **Temperatura do Modelo:** Ajuste a temperatura do modelo para controlar a criatividade. Utilize valores baixos para respostas mais factuais.
- **Consistência Própria:** Gere múltiplas respostas e peça ao modelo para desenvolver um consenso.
- **Delimitadores**: Utilize delimitadores como `--` ou tags XML (`<tag>conteúdo</tag>`) para separar seções do prompt, como instruções, exemplos e dados. Isso ajuda o modelo a distinguir diferentes partes do prompt.
- **Variáveis**: Ao usar plataformas como o console da Anthropic, você pode definir variáveis no prompt usando a sintaxe `{{variável}}`. Isso permite que você crie prompts genéricos que podem ser facilmente adaptados para casos específicos, substituindo apenas os valores das variáveis.

### Ferramentas 

- Testar Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Gerador de Prompts https://www.feedough.com/ai-prompt-generator/

## Repository Bibliographic References:
- https://www.promptingguide.ai/

- Large Language Models are Zero-short Reasoners Contrastive Chin of Thought prompting
- Guiding Large Language Models via Directional Stimulus Prompting
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Deliberate Problem Solving with Large Language Models
- Meta Prompting for AI Systems
- Generative Agents: Interactive Simulacra of Human Behavior
- Large Language Model Guided Tree-of-Thought
- Skeleton of Thought: Augmenting Language Models with Variable-Depth Reasoning Ability
-  Generated Knowledge Prompting for Commonsense Reasoning
- Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations
- Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks
- PAL: Program-aided Language Models
- Synergizing Reasoning and Acting in Language Models






- **Markdown**: Use formatação Markdown para estruturar seus prompts, incluindo títulos (# ## ###), negrito (`*texto**`), itálico (`texto*`), listas (`item`) e blocos de código (````código````). Isso melhora a legibilidade para humanos e ajuda o modelo a entender a estrutura do prompt.

