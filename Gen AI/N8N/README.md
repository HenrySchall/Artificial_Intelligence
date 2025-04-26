# Large Language Models (LLM's)

> As LLM's são modelos de AI, baseados em técnicas de Machine Learning de aprendizado não-supervisionado, ou seja, são modelos alimentados por um conjunto de dados (centenas de palavras e frases) os quais são extraidos informações desses dados para se criar conexões e "aprender" sem qualquer instrução humana, como resultado ele captura as relações complexas entre palavras e frases dando origem ao processo de Natural Language Processing (NLP). As LLM's requerem recursos computacionais significativos para processar dados, por isso utilizam unidades de processamento gráfico (GPUs), para acelerar o treinasmento e a opoeração dos transformers.

![userlmn_61d38ca6d280b8867ccc36047a493b56](https://github.com/user-attachments/assets/e3447078-f291-4ded-8621-2165489ca052)

Source: Nvidia

> Trnasformers são uma arquitetura de Deep Learning (redes neurais profundas) que transformam ou alteram uma sequência de entra em uma sequência de saída, foi apresentada pelo paper "Attention is All you Need". Sua arquitetura melhora a acapacidade dos modelos de Machine Learning ao capturar dependências contextuais em sequências de dados, como palavras em uma frase. As arquiteturas anteriores como RNN e LSTM perdiam detalhes em sequências longas, por se basearem em codificado e decodificado, a solução dos transformers introduzem mecanismo de atenção que capturam as relações contextuais entre todas as partes de um sequencia simultanemante ou seja em vez de processa sequências passo a passo (um elemento da sequência por vez). entendem não processam sequencialmente — olham toda a sequência de uma vez. entenem o contexto Porta	Função
Forget gate	Decide o que esquecer do estado de célula anterior.
Input gate	Decide o que adicionar de novo à memória.

Os LLMs são uma classe dos chamados Foundation Models (Termo popularizado plea Universidade de Stanford em 2021. Enquanto LLMs são modelos especificos para lingiagem portando focam em tarefas relacionadas a texto (Prompt Engineering) os foudation Models tem um escopo mais amplo podendo ser apliucados a ddifversa finalidades como aplicacoes envolveno aimgfe audio video e etc

>
>São modelos de Inteligência Aritificial (AI) que utilizam técnicas de Machine Learning para entender e gerar linguagem humana, utilizando Processamento de Linguagem NAtural (PLN) para interarigr com a linguagem humana
## Prompt Engineering

https://www.promptingguide.ai/introduction

- Testar Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- https://lmarena.ai/
- https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
- Gerador de Prompts https://www.feedough.com/ai-prompt-generator/

> A engenharia de prompt é a ciência empírica de planejar, criar e testar prompts para gerar melhores respostas em grandes modelos de linguagem.

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

### Técnicas 

- **Zero-shot**: Em um prompt zero-shot, você fornece uma instrução direta para o modelo, sem incluir exemplos. Isso é útil para tarefas simples e diretas, onde o modelo pode gerar uma resposta adequada sem a necessidade de exemplos adicionais.
  
- **Estímulo Direcional:** Você pode colocar algumas “dicas” ou keywords para guiar o modelo melhor. Assim facilita a ter o resultado esperado com algumas poucas palavras.
    
- **Few-shot**: Few-shot prompting envolve fornecer alguns exemplos (geralmente entre 1 e 5) de entradas e saídas desejadas no prompt. Isso ajuda o modelo a entender melhor a tarefa e gerar respostas mais precisas e consistentes. Os exemplos devem ser escolhidos cuidadosamente para serem representativos da tarefa.

- **Chain-of-Thought (CoT)**: A técnica Chain-of-Thought (Cadeia de Pensamento) envolve fazer com que o modelo explique seu raciocínio passo a passo antes de chegar à resposta final. Isso é especialmente útil para tarefas que exigem raciocínio lógico ou resolução de problemas. Ao fornecer exemplos de como o raciocínio deve ser explicado e solicitar que o modelo siga o mesmo processo, você pode melhorar a qualidade e a precisão das respostas.

- **Self-Consistency**: A técnica de Self-Consistency envolve gerar múltiplas cadeias de pensamento para a mesma tarefa e, em seguida, fazer com que o modelo escolha a resposta mais consistente. Isso ajuda a reduzir erros e melhorar a qualidade das respostas, aproveitando a capacidade do modelo de avaliar seu próprio raciocínio.

- **Tree-of-Thought (ToT)**: A técnica Tree-of-Thought expande a ideia da Chain-of-Thought, gerando múltiplos pensamentos e desenvolvendo uma árvore de raciocínio. O modelo explora diferentes caminhos de raciocínio e escolhe o mais promissor para chegar à resposta final. Isso é útil para problemas complexos que podem ter várias abordagens possíveis.

- **Skeleton-of-Thought (SoT)**: A técnica Skeleton-of-Thought envolve gerar um esqueleto ou índice de tópicos antes de desenvolver o conteúdo completo. Isso ajuda a estruturar a resposta e garantir que todos os pontos-chave sejam abordados. O modelo primeiro gera o esqueleto e, em seguida, preenche cada tópico com detalhes.
    
- **Generated Knowledge Prompting**: Essa técnica envolve usar o modelo de linguagem para gerar conhecimento contextual adicional que pode ser usado para melhorar a qualidade das respostas. O modelo gera informações relevantes com base no contexto fornecido, que são então incorporadas ao prompt para ajudar a gerar respostas mais precisas e informativas.

- **Prompt Maiêutico:** A técnica do Prompt Maiêutico envolve pedir ao modelo para justificar suas respostas, explicando o raciocínio por trás delas. Isso pode ajudar a melhorar a qualidade das respostas, incentivando o modelo a fornecer explicações mais detalhadas e lógicas.

- **Retrieval Augmented Generation (RAG)**: A técnica RAG combina modelos de linguagem com bases de conhecimento externas para gerar respostas mais precisas e informativas. O modelo recupera informações relevantes da base de conhecimento e as utiliza para complementar seu próprio conhecimento ao gerar a resposta.

- **PAL (Program-Aided Language Models)**: A técnica PAL envolve usar conceitos e estruturas de linguagens de programação, como variáveis e funções, dentro dos prompts. Isso pode ajudar a tornar os prompts mais modulares, reutilizáveis e fáceis de adaptar para diferentes casos de uso.

- **ReAct (Reason + Act)**: A técnica ReAct divide tarefas complexas em etapas de raciocínio e ação. O modelo primeiro raciocina sobre a tarefa, decidindo qual ação tomar, e depois executa essa ação. Esse processo é repetido até que a tarefa seja concluída. Isso é especialmente útil para tarefas que exigem várias etapas ou interação com ferramentas externas.

### Repository Bibliographic References:
- Large Language Models are Zero-short Reasoners Contrastive Chin of Thought prompting, by Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa. Available: https://arxiv.org/abs/2205.11916

- Guiding Large Language Models via Directional Stimulus Prompting, by Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan. Available: https://arxiv.org/abs/2302.11520

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, by Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan. Available: https://arxiv.org/abs/2201.11903

- Self-Consistency Improves Chain of Thought Reasoning in Language Models, by Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. Available: https://arxiv.org/abs/2203.11171

- Deliberate Problem Solving with Large Language Models, by Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan. Available: https://arxiv.org/abs/2305.10601

- Generative Agents: Interactive Simulacra of Human Behavior, by Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. Available: https://arxiv.org/abs/2304.03442

- Large Language Model Guided Tree-of-Thought, by Jieyi Long. Available: https://arxiv.org/abs/2305.08291

- Skeleton of Thought: Augmenting Language Models with Variable-Depth Reasoning Ability, by Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, Yu Wang. Available: https://arxiv.org/abs/2307.15337

-  Generated Knowledge Prompting for Commonsense Reasoning, by Jiacheng Liu, Alisa Liu, Ximing Lu, Sean Welleck, Peter West, Ronan Le Bras, Yejin Choi, Hannaneh Hajishirzi. Available: https://arxiv.org/abs/2110.08387

- Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations, by Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, Yejin Choi. Available: https://arxiv.org/abs/2205.11822

- Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks, by Katja Filippova. Available: https://aclanthology.org/2020.findings-emnlp.76/

- PAL: Program-aided Language Models. by Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig. Available: https://arxiv.org/abs/2211.10435

- Synergizing Reasoning and Acting in Language Models, by Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao. Available: https://arxiv.org/abs/2210.03629

https://arxiv.org/abs/1706.03762

