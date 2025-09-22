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
"yfinance", "scikit-learn", "pyomo", "panel", "hvplot", "holoviews", "datashader", "param", "colorcet",
"transformers","einops","accelerate", "bitsandbytes", "torch", "torchvision","torchaudio"]

install_packages(packages_list)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB as GNB

# License Key Gurobi -> grbgetkey 6b409973-a9d4-42f8-86ef-0bca3652fa1d

#################
### Example 1 ###
#################

x1 = gp.addVar()
x2 = gp.addVar()
x3 = gp.addVar()

m.setObjective

# Função Objeto
gp.setObjective(12*x1 + 18*x2 + 22*x3, sense = gp.GRB.Maximize)

# Restrições
y1 = gp.addConstr(1.5*x1 + 1.2*x3 <= 120)
y2 = gp.addConstr(2.2*x1 + 1.4*x3 <= 200)
y3 = gp.addConstr(1.2*x1 + 2.0*x2 + 2.4*x3 <= 250)

m.optimize

# Combinação ideal de produtos
print("Produto A:", x1.X)
print("Produto B:", x2.X)
print("Produto C:", x3.X)

# Tempo de uso da máquina ideal de produtos
tempo_m1 = 120 - c1.Slack
tempo_m2 = 200 - c2.Slack
tempo_m3 = 250 - c3.Slack

print("Máquina_1", tempo_m1)
print("Máquina_2", tempo_m2)
print("Máquina_3", tempo_m3)

#################
### Example 2 ###
#################

itens = 15
capacidade = 1000

# Rotulagem
itens = list()
for i in range (itens):
    rotulo = "Item_{}".format(i+1)
    itens.append(rotulo)

vetor_pesos = [65,94,119,59,149,114,57,136,100,150,122,117,120,130,133]
valores = dict()
for index, valor in enumerate(vetor_pesos):
    rotulo = itens[index]
    valores[rotulo] = valor

vetor_pesos = [455,691,833,425,1064,758,419,914,651,966,828,827,857,837,894]
valores = dict()
for index, valor in enumerate(vetor_pesos):
    rotulo = itens[index]
    valores[rotulo] = valor

print(valores)

# Função Objeto
x = gp.addVars(itens, vtype=gp.GBR.BINARY)
gp.setObjective(gp.quicksum(x[i] * valores[i] for i in itens), sense = gp.GRB.Maximize)

# Restrição
c1 = gp.addConstr(gp.quicksum(x[i] * pesos[i] for i in itens) <= capacidade)

gp.optimize()

for item in itens:
    print(item, round(x[item].X))