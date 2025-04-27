########################
### Install Packages ###
########################

import subprocess
import sys

def install_packages(pacotes):
    for pacote in pacotes:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

# List of packages
packages_list = ["radian", "numpy", "pandas", "matplotlib", "scipy", "seaborn","statsmodels", "plotly", "gurobipy",
"yfinance", "scikit-learn", "pyomo", "panel", "hvplot", "holoviews", "datashader", "param", "colorcet",
"transformers","einops","accelerate", "bitsandbytes", "torch", "torchvision","torchaudio"]

install_packages(packages_list)

#####################
### Load Packages ###
#####################

import radian as rd
import pyomo.environ as pyo
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB as GNB

##################
### Introdução ###  
##################

# Install Cuda Toolkit = https://developer.nvidia.com/cuda-toolkit
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

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
# tokenizer=tokenizer: especifica o tokenizador que o pipeline usará. Passamos o tokenizer que carregamos anteriormente para garantir que o texto de entrada seja tokenizado corretamente e os tokens de saída sejam decodificados com precisão.~~