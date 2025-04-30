# Prompt Engineering

> Engenharia de prompt é a prática de projetar, testar e refinar entradas (prompts) para obter modelos de linguagem (LLM's). Pesquisadores utilizam a engenharia de prompts para aprimorar a capacidade de LLMs em uma ampla gama de tarefas e desenvolvedores utilizam a engenharia de prompts para projetar técnicas de prompts robustas e eficazes para interagir com LLMs e outras ferramentas. Sendo assim, podemos dizer que é o desenvolvimento de prompts mais precisos, relevantes e controláveis.

## Estrutura básica de prompt (P.R.O.M.P.T.)

- **Persona**: Defina a persona ou papel que o modelo deve assumir ao responder.
- **Roteiro**: Especifique o roteiro ou tarefa que o modelo deve realizar.
- **Objetivo**: Deixe claro qual é o objetivo do prompt e o que se espera alcançar.
- **Modelo**: Indique o formato ou estrutura que a resposta do modelo deve seguir.
- **Panorama**: Forneça informações contextuais relevantes para ajudar o modelo a gerar uma resposta mais precisa e específica.
- **Transformar**: Esteja preparado para refinar e iterar o prompt com base nos resultados obtidos, fornecendo feedback e exemplos adicionais se necessário.

## Evitando alucinações

- Permitir que o modelo diga que não sabe a resposta: Ao dar permissão explícita para o modelo admitir quando não tem conhecimento suficiente para responder uma pergunta, você pode reduzir a quantidade de informações fabricadas ou imprecisas nas respostas.
- Pedir para o modelo encontrar evidências antes de responder: Instrua o modelo a procurar trechos relevantes no contexto fornecido que embasem sua resposta. Você pode usar tags especiais, como `<Quotes>`, para indicar onde o modelo deve citar as evidências encontradas.
    - Exemplo prático: No vídeo, é mostrado um exemplo usando tags `<Quotes>` para fazer o modelo buscar citações relevantes no texto fornecido antes de gerar sua resposta.
- Permitir que o modelo "pense" antes de responder: Dê espaço para o modelo refletir e processar a informação antes de gerar uma resposta.
- **Temperatura do Modelo:** Ajuste a temperatura do modelo para controlar a criatividade. Utilize valores baixos para respostas mais factuais.
- **Consistência Própria:** Gere múltiplas respostas e peça ao modelo para desenvolver um consenso.
- **Markdown**: Use formatação Markdown para estruturar seus prompts, incluindo títulos (# ## ###), negrito (`*texto**`), itálico (`texto*`), listas (`item`) e blocos de código (````código````). Isso melhora a legibilidade para humanos e ajuda o modelo a entender a estrutura do prompt.
- **Delimitadores**: Utilize delimitadores como `--` ou tags XML (`<tag>conteúdo</tag>`) para separar seções do prompt, como instruções, exemplos e dados. Isso ajuda o modelo a distinguir diferentes partes do prompt.
- **Variáveis**: Ao usar plataformas como o console da Anthropic, você pode definir variáveis no prompt usando a sintaxe `{{variável}}`. Isso permite que você crie prompts genéricos que podem ser facilmente adaptados para casos específicos, substituindo apenas os valores das variáveis.





- Testar Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Gerador de Prompts https://www.feedough.com/ai-prompt-generator/

### Repository Bibliographic References:
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
