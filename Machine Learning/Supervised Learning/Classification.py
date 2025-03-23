import string as string
import random as random
import radian as rd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

####################
### Introduction ###
####################

dados = "https://drive.google.com/uc?export=download&id=1wMapByTvMFt16zz9Bd2643eTHJXtEhnX"
df = pd.read_csv(dados)

df
df.describe()

df[df['income'] >= 69995.685578]
df[df['loan'] <= 1.377630]