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
import gc
import param
import sklearn
import scipy
import string
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

#########################
### Foundation Models ###
#########################

###################
### HuggingFace ###
###################

# My License Keys
# https://drive.google.com/file/d/1aMw7MGhE8FOPBs5iUWAvknoeKtth2ooZ/view?usp=drive_link

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(20)
os.environ["HF_TOKEN"] = "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"

id_model = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")

### Tokenizar & Pipeline ###

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.1, "do_sample": True,}

# Example 1
prompt = "Quanto é 7 x 6 - 42?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# Example 2
prompt = "Explique o que é computação quântica"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# Limpar cache
# gc.collect()
# torch.cuda.empty_cache()
# print("Cache de memória CUDA esvaziado.")

### Templates ###
# https://huggingface.co/docs/transformers/chat_templating

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

# Example 3
prompt = "O que é IA?" 

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

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

##############
### CrewAI ###
##############