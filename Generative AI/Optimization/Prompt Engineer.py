########################
### Install Packages ###
########################

import subprocess
import sys

def install_packages(pacotes):
    for pacote in pacotes:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

# List of packages
packages_list = ["numpy", "pandas", "matplotlib", "scipy", "seaborn","statsmodels", "plotly", "gurobipy",
"yfinance", "scikit-learn", "panel", "datashader", "param", "colorcet", "transformers","einops","accelerate", 
"bitsandbytes"]

install_packages(packages_list)

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# print(torch.__version__)
# pip install cuda-python==12.8
# print(torch.version.cuda)

# Links 
# - https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html#f1
# - https://pytorch.org/get-started/locally/

#####################
### Load Packages ###
#####################

import gurobipy as gp
import pandas as pd 
import seaborn as sns
import plotly.express as px
import numpy as np
import panel as pn 
import seaborn.objects as so
import matplotlib as mpl
import colorcet as cc
import matplotlib.pyplot as plt
import math
import datetime
import param
import sklearn
import scipy
import string
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

#################
### Langchain ###
#################

# pip install langchain, langchain-community, langchain-huggingface, langchainhub, langchain_chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import (HumanMessage, SystemMessage)
from langchain_huggingface import ChatHuggingFace










# Carregando LLM via pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) # o que é quantização

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model = model, tokenizer = tokenizer, task = "text-generation", temperature = 0.1,
max_new_tokens = 500, do_sample = True, repetition_penalty = 1.1, return_full_text = False)

llm = HuggingFacePipeline(pipeline = pipe)

input = "Quem foi a primeira pessoa no espaço?"

output = llm.invoke(input)
print(output)

del_id = "meta-llama/Meta-Llama-3-8B-Instruct"

template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

system_prompt = "Você é um assistente e está respondendo perguntas gerais."
user_prompt = input

prompt_template = template.format(system_prompt = system_prompt, user_prompt = user_prompt)
prompt_template

output = llm.invoke(prompt_template)
output

# Modelos de Chat

sgs = [
    SystemMessage(content = "Você é um assistente e está respondendo perguntas gerais."),
    HumanMessage(content = "Explique para mim brevemente o conceito de IA.")
]

hat_model = ChatHuggingFace(llm = llm)

odel_template = tokenizer.chat_template
model_template

#chat_model._to_chat_prompt(msgs)

#res = chat_model.invoke(msgs)
#print(res.content)

## Langchain

> É uma biblioteca de código aberto projetada para facilitar a integração de LLM's como

> Modelos de LLM's disponíveis https://python.langchain.com/v0.2/docs/integrations/llms/

> Devido ao fato do LangChain se integrar bem ao Hugging Face, podemos usar a mesma lógica de pipeline que vimos antes na implementação com o Transformers. São uma ótima e fácil maneira de usar modelos para inferência.quantização e também a tokanização.












Componentes
- Modelos
- Prompts
- Chains = encaderar LLM's m aplicações complexas, permitindo a ligação entre múltiplos modelos ou outros módulos especializados
- Memória: Módulos que permitem o gerenciamento e alteração de conversas anteriores, essencial para chatbots que preciam relembrar interações passadas para manter coerência
- agentes = euqipads com kit de ferramentas abrangentes, que possibvvilita escolher quais ferramentas usar com base nas informações do usuário
- Indices = metod spara organizar documentos 

Econosistema
- langchain community - integraç~~oes com terceiros tipo langchain-openai
- lanchiga chain = chains, agentes e estrategias de retrivel quwe compoem a arquiteutra cognitiva de uma aplicação 
- langraph: para construir aplicações robustas e como estado para múltiplos atores com LLM's, modelando etapas como arestas e nós em um gráfico. Integra-se perfeitamente com Langchain, mas pode ser usado sem ele
- langserve: Para implementar chains do lang cghain com oapis rest 
- langsmith platadfoma para desenvolver de aplicacoes LLM

## MCP
## RAG