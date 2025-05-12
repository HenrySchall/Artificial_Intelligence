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

#####################
### Configuration ###
#####################

# My License Keys
# https://drive.google.com/file/d/1aMw7MGhE8FOPBs5iUWAvknoeKtth2ooZ/view?usp=drive_link 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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(20)
os.environ["HF_TOKEN"] = "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"

id_model = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.1, "do_sample": True}

# Limpar cache
# gc.collect()
# torch.cuda.empty_cache()
# print("Cache de memória CUDA esvaziado.")

#######################
### Prompt Engineer ###
#######################

# Example 1
prompt = "Gere um código em python que escreva a sequência de fibonnaci"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

# Example 2
prompt = "Gere um código em python que escreva a sequência de fibonnaci"

sys_prompt = "Você é um programador experiente. Retorne o código requisitado e forneça explicações breves se achar conveniente"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

# Tesntando o resultado 
def fibonacci(n):
    a, b = 0, 1
    sequence = []
    while len(sequence) < n:
        sequence.append(a)
        a, b = b, a + b
    return sequence

n = 10  
print(fibonacci(n))

### Formatando Mensagem ###

# Example 1
prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

messages = [
    {"role": "system", "content": prompt_sys},
    {"role": "user", "content": prompt}
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

# Example 2 
id_model = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
generation_args = {"max_new_tokens": 100, "do_sample": True, "temperature": 0.7, "return_full_text": False}

prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": prompt_sys},
    {"role": "assistant", "content": "Claro! Aqui estão: Paris, Roma, Londres, Berlim, Madri, Viena, Amsterdã, Lisboa, Praga e Barcelona."},
    {"role": "user", "content": "Qual delas é a melhor para visitar em dezembro?"},
    {"role": "assistant", "content": ""}  # Modelo completará a próxima resposta
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

####################
### Quantization ###
####################

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

Para aplicar a quantização, agora carregaremos o modelo com o método "AutoModelForCausalLM", conforme citado anteriormente

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = ("Quem foi a primeira pessoa no espaço?")
messages = [{"role": "user", "content": prompt}]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
generated_ids = model.generate(model_inputs, max_new_tokens = 1000, do_sample = True,
                               pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
res = decoded[0]
res