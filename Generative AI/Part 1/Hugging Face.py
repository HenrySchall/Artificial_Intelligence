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
import getpass
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

####################
### Hugging Face ###
####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# torch.random.manual_seed()
os.environ["HF_TOKEN"] = getpass.getpass()

id_model = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(id_model, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")

######################################
### Tokenize and Prompt Generation ###
######################################

tokenizer = AutoTokenizer.from_pretrained(id_model)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.1, "do_sample": True}

# Example 1
prompt = "Quanto é 7 x 7 - 36?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# Example 2
prompt = "Quem foi a primeira pessoa no espaço?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# Para descobrir o template adequado, sempre cheque a descrição do modelo, por exemplo: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt) # .format para concatenar o prompt nesse template, assim não precisamos redigitar ali manualment
template

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








template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

template
output = pipe(template, **generation_args)
print(output[0]['generated_text'])

prompt = "O que é IA?"  # @param {type:"string"}

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

#  Explorando mais prompts

#prompt = "O que é IA? "  # @param {type:"string"}
#prompt = "O que é IA? Responda em 1 frase" # @param {type:"string"}
prompt = "O que é IA? Responda em forma de poema" # @param {type:"string"}

sys_prompt = "Você é um assistente virtual prestativo. Responda as perguntas em português."

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

print(template)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

prompt = "Gere um código em python que escreva a sequência de fibonnaci"

sys_prompt = "Você é um programador experiente. Retorne o código requisitado e forneça explicações breves se achar conveniente"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(sys_prompt, prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])


# Melhorando qualidade 

https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct

# Formatando mensagens

prompt = "O que é IA?"

msg = [
    {"role": "system", "content": "Você é um assistente virtual prestativo. Responda as perguntas em português."},
    {"role": "user", "content": prompt}
]

output = pipe(msg, **generation_args)
print(output[0]["generated_text"])

prompt = "Liste o nome de 10 cidades famosas da Europa"
prompt_sys = "Você é um assistente de viagens prestativo. Responda as perguntas em português."

msg = [
    {"role": "system", "content": prompt_sys},
    {"role": "user", "content": prompt},
]

output = pipe(msg, **generation_args)
print(output[0]['generated_text'])

# Checar GPU -> !nvidia-smi

# otimização

https://huggingface.co/blog/4bit-transformers-bitsandbytes
https://github.com/AutoGPTQ/AutoGPTQ
https://github.com/casper-hansen/AutoAWQ

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

https://huggingface.co/docs/transformers/chat_templating

prompt = ("Quem foi a primeira pessoa no espaço?")
messages = [{"role": "user", "content": prompt}]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
model_inputs = encodeds.to(device)
generated_ids = model.generate(model_inputs, max_new_tokens = 1000, do_sample = True,
                               pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
res = decoded[0]
res

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

##############
### CrewAI ###
##############