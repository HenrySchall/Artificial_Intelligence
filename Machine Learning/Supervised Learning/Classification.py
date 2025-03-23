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

# Retiado de: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
dados = "https://drive.google.com/uc?export=download&id=1wMapByTvMFt16zz9Bd2643eTHJXtEhnX"
df = pd.read_csv(dados)

df
df.describe()

# paid the loan
# didn't pay the loan

max_val = df["income"].max()
print(max_val)

min_val = df["loan"].min()
print(min_val)

df[df["income"] >= 69995.6855783239]
df[df["loan"] <= 1.4000000]

plt.hist(x = df['age'])
plt.title("Distribuição dos Valores")
plt.show()

sns.countplot(x = df['default'])
plt.title("Distribuição dos Valores")
plt.show()

grafico = px.scatter_matrix(df, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()

# The presence of negative ages is observed...

#################
### Treatment ###
#################

df.loc[df['age'] < 0]

base_credit2 = df.drop('age', axis = 1)
base_credit2