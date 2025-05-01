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
- Leitura = https://www.promptingguide.ai/
- Testar Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Gerador de Prompts https://www.feedough.com/ai-prompt-generator/

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


