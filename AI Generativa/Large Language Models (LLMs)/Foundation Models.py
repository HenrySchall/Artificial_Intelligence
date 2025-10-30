#############################
### Configurando Ambiente ###
#############################

# Links 
# - https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html#f1
# - https://pytorch.org/get-started/locally/

# pip install transformers==4.48.2 einops accelerate bitsandbytes
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
print(torch.__version__)

# pip install cuda-python == 12.8
print(torch.version.cuda) 

import pandas as pd 
import seaborn as sns
import plotly.express as px
import numpy as np
import panel as pn 
import seaborn.objects as so
import matplotlib as mpl
import colorcet as cc
import matplotlib.pyplot as plt
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

#########################
### Foundation Models ###
#########################

# Verificando se a TPU/GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Definido seed e API Key 
torch.random.manual_seed(42)
os.environ["HF_TOKEN"] = "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"

#################
### Example 1 ###
#################

id_model = "microsoft/Phi-3-mini-4k-instruct"
modelo = AutoModelForCausalLM.from_pretrained(
    id_model, 
    device_map = "cuda", 
    torch_dtype = "auto", 
    trust_remote_code = True, 
    attn_implementation= "eager")

tokenizar = AutoTokenizer.from_pretrained(id_model)

pipe = pipeline(
    "text-generation", 
    model = modelo, 
    tokenizer = tokenizar)

generation_args = {
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.1, 
    "do_sample": True}

prompt = "Explique o que é computação quântica"
output = pipe(prompt, **generation_args)
print(output[0]["generated_text"])

#################
### Example 2 ###
#################

id_model = "microsoft/Phi-3-mini-4k-instruct"
modelo = AutoModelForCausalLM.from_pretrained(
    id_model, 
    device_map = "cuda", 
    torch_dtype = "auto", 
    trust_remote_code = True, 
    attn_implementation= "eager")

tokenizar = AutoTokenizer.from_pretrained(id_model)

pipe = pipeline(
    "text-generation", 
    model = modelo, 
    tokenizer = tokenizar)

generation_args = {
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.1, 
    "do_sample": True,
    "repetition_penalty": 1.1}

# Example 2
prompt = "Quanto é 7 x 6 - 42?"
output = pipe(prompt, **generation_args)
print(output[0]["generated_text"])

##################
### Tratamento ###
##################

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model=model, tokenizer=tokenizer, task="text-generation", temperature=0.1, max_new_tokens=500, do_sample=True, repetition_penalty=1.1, return_full_text=False)
llm = HuggingFacePipeline(pipeline=pipe)

input = "Qual foi a primeira linguagem de programação?"
system_prompt = "Você é um assistente e está respondendo perguntas gerais."

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

prompt_template = template.format(system_prompt = system_prompt, user_prompt = input)
prompt_template

output = llm.invoke(prompt_template)
print(output)

#######################
### Modelos de Chat ### https://python.langchain.com/v0.2/docs/integrations/chat/

msgs = [
    SystemMessage(content = "Você é um assistente e está respondendo perguntas gerais."),
    HumanMessage(content = "Explique para mim brevemente o conceito de IA.")
]

chat_model = ChatHuggingFace(llm = llm)

model_template = tokenizer.chat_template
model_template

chat_model._to_chat_prompt(msgs)

res = chat_model.invoke(msgs)
print(res.content)

# Example 1
prompt = "Gere um código em python que escreva a sequência de fibonnaci"

template = """<|system|>
{}<|end|>
<|user|>
"{}"<|end|>
<|assistant|>""".format(prompt)

output = pipe(template, **generation_args)
print(output[0]['generated_text'])










































#Limpar cache casos esteja travando o deploy
#gc.collect()
#torch.cuda.empty_cache()
#print("Cache de memória CUDA esvaziado.")

################# 
### Langchain ###
#################

# pip install langchain, langchain-community, langchain-huggingface, langchainhub, langchain_chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import (HumanMessage, SystemMessage)
from langchain_huggingface import ChatHuggingFace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(20)
os.environ["HF_TOKEN"] = "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"

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "cuda", torch_dtype = "auto", trust_remote_code = True, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(model = model, tokenizer = tokenizer, task = "text-generation", temperature = 0.1, max_new_tokens = 500, do_sample = True, repetition_penalty = 1.1, return_full_text = False)
llm = HuggingFacePipeline(pipeline = pipe)

input = "Quem foi a primeira pessoa no espaço?"
output = llm.invoke(input)
print(output)

##############
### CrewAI ###
##############

####################
### Quantization ###
####################

model_id = "microsoft/Phi-3-mini-4k-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens = 500,
    do_sample = True,
    repetition_penalty = 1.1,
    return_full_text = False,
)

llm = HuggingFacePipeline(pipeline = pipe)

input = "Quem foi a primeira pessoa no espaço?"
output = llm.invoke(input)
print(output)

