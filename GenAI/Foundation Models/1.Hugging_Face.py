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
# pip install cuda-python==12.8

# Links 
# - https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html#f1
# - https://pytorch.org/get-started/locally/

# Uninstall Packages
# pip freeze > packages.txt
# pip uninstall -y -r packages.txt

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

##################
### Introdução ###  
##################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print(torch.__version__) # Versão PyTorch
print(torch.version.cuda) # Versão Cuda    

torch.random.manual_seed(42)
os.environ["HF_TOKEN"] = "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"

id_model = "microsoft/Phi-3-mini-4k-instruct"

# 4k = context window size or token sequence length (4000 mil tokens em uma unica entrada, gerar sequencias de texto)
# instruct = model type

model = AutoModelForCausalLM.from_pretrained(id_model,
    device_map = "cuda", # processamento usando a placas de vídeo (GPU) + processador (CPU) - Tecnologia NVIDIA
    torch_dtype = "auto",
    trust_remote_code = True,
    attn_implementation="eager")

# device_map="cuda" = Especifica que o modelo deve ser processado usando GPU + CPU
# torch_dtype="auto" = Define automaticamente o tipo de dados apropriado para os tensores do modelo
# trust_remote_code=True = Permite o carregamento de código personalizado do repositório de modelos no HuggingFace
# attn_implementation="eager" = Especifica o método de implementação para o mecanismo de Self-Attention. Sendo a configuração "eager" é uma implementação 
# particular que fornecer melhor desempenho  para alguns modelos ao processar o mecanismo de Self-Attention

# Tokenizar

# Em nossa configuração, também precisamos carregar o tokenizer associado ao modelo. O tokenizer é crucial para preparar dados de texto em um formato que o modelo possa entender.
# Um tokenizador converte texto bruto em tokens

tokenizer = AutoTokenizer.from_pretrained(id_model)

# Criação de Pipiline
# gora criaremos um pipeline para geração de texto usando nosso modelo e tokenizer carregados anteriormente. A função de pipeline HuggingFace 
# simplifica o processo de execução de várias tarefas de processamento de linguagem natural ao fornecer uma interface de alto nível. Um pipeline é uma abstração que simplifica o uso de modelos 
# pré-treinados para uma variedade de tarefas de PLN. Ele fornece uma API unificada para diferentes tarefas, como geração de texto, classificação de texto, tradução e muito mais.

pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

# "text-generation": especifica a tarefa que o pipeline está configurado para executar. Neste caso, estamos configurando um pipeline para geração de texto. O pipeline usará o modelo para gerar texto com base em um prompt fornecido.
# model=model: especifica o modelo pré-treinado que o pipeline usará. Aqui, estamos passando o model que carregamos anteriormente. Este modelo é responsável por gerar texto com base nos tokens de entrada.
# tokenizer=tokenizer: especifica o tokenizador que o pipeline usará. Passamos o tokenizer que carregamos anteriormente para garantir que o texto de entrada seja tokenizado corretamente e os tokens de saída sejam decodificados com precisão.

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1, # 0.1 até 0.9
    "do_sample": True,
}


# max_new_tokens" -> numero maxim ode token que devem ser gerados (comprimento do texto)
# "return_full_text": False, (se deve retornar o texto completo)
#  "temperature": 0.1, # 0.1 até 0.9 (aleatoriade do processo = grau de criativade)
# "do_sample": True, (amostragem true = com base nas probalibdades e false e escolhe o token  de maior probabilidade)

prompt = "Quanto é 7 x 6 - 42?"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# **generation_args: Isso descompacta o dicionário generation_args e passa seu conteúdo como argumentos de palavra-chave para o pipeline, personalizando o processo de geração de texto.

prompt = "Explique o que é computação quântica"
output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

# Templates e engenharia de prompt

# Essas tags formadas por <|##nome##|> são o que chamamos de Tokens especiais (special tokens) e são usadas para delimitar o início e fim de texto e dizer ao modelo como queremos que a mensagem seja interpretada
# Os tokens especiais usados para interagir com o Phi 3 são esses:
# <|system|>, <|user|> e <|assistant|>: correspondem ao papel (role) das mensagens. Os papéis usados aqui são: system, user e assistant
# <|end|>: Isso é equivalente ao token EOS (End of String), usado para marcar o fim do texto/string.
# Usaremos o .format para concatenar o prompt nesse template, assim não precisamos redigitar ali manualmente

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