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

