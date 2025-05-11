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


# RAG

# MCP

