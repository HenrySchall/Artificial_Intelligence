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

##########################
### RAG - Hugging Face ###
##########################

# pip install langchain, langchain-community, langchain-huggingface, langchainhub, langchain_chroma

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# torch.random.manual_seed(42)
os.environ["HF_TOKEN"] = "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"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model=model, tokenizer=tokenizer, task="text-generation", temperature=0.1, max_new_tokens=500, do_sample=True, repetition_penalty=1.1, return_full_text=False,)
llm = HuggingFacePipeline(pipeline=pipe)

# Templates

# PHI 3
#template = """
#<|system|>
#Você é um assistente virtual prestativo e está respondendo perguntas gerais. <|end|>
#<|user|>
#{pergunta}<|end|>
#<|assistant|>
#"""

# LLAMA 3
template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Você é um assistente virtual prestativo e está respondendo perguntas gerais.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{pergunta}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

template


prompt = PromptTemplate.from_template(template)
prompt

###########
### MCP ###
###########