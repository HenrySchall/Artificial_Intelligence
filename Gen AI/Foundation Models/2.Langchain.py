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
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

##################
### Introdução ###
##################

# pip install langchain, langchain-community, langchain-huggingface, langchainhub, langchain_chroma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Carregando LLM via pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model = model, tokenizer = tokenizer, task = "text-generation", temperature = 0.1, 
max_new_tokens = 500, do_sample = True, repetition_penalty = 1.1, return_full_text = False,)

# task - Este parâmetro corresponde à tarefa que o modelo está desempenhando. Mencionamos a tarefa para ser geração de texto ("text-generation")
# temperature = controle de aleatoriedade, quanto mais baixo, mais deterministica e quanto mais alto, mas criativa
# repetition_penalty — O parâmetro para Penalidade de repetição é um fator aplicado para desencorajar o modelo de gerar texto ou frases repetitivas. Ao ajustar essa penalidade você pode influenciar a saída do modelo, reduzindo a probabilidade de ele produzir conteúdo redundante ou repetido. 1.0 significa nenhuma penalidade (1.0 é o valor padrão)
# max_new_tokens = 
# do_sample = adicvionar um fator aleatorio
# repetition_penalty = 1.1 (reduncacia 1= padrao acima e penalidade)

llm = HuggingFacePipeline(pipeline = pipe)
input = "Quem foi a primeira pessoa no espaço?"
output = llm.invoke(input)
print(output)

# Adequando o prompt 
# <|begin_of_text|>: equivalente ao token BOS (Beginning of String), indicando o início de uma nova sequência de texto.
# <|eot_id|>: indica o fim de uma mensagem.
# <|start_header_id|>{role}<|end_header_id|>: esses tokens envolvem o papel de uma mensagem específica. Os papéis possíveis são: system, user e assistant.
# <|end_of_text|>: Isso é equivalente ao token EOS (End of String). Ao chegar nesse token, o Llama 3 deixará de gerar mais tokens.