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

### Tools
- Guide = https://www.promptingguide.ai/
- Test Prompts = https://app.chathub.gg/?utm_source=chathub.gg
- Generator of Prompts https://www.feedough.com/ai-prompt-generator/

### Techniques


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


