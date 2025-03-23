import string as string
import random as random
import radian as rd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

####################
### Introduction ###
####################

# Taken from: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
dados = "https://drive.google.com/uc?export=download&id=1wMapByTvMFt16zz9Bd2643eTHJXtEhnX"
df = pd.read_csv(dados)

df
df.describe()

# 1 = paid the loan
# 0 = didn't pay the loan

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

# Inconsistent Values
df.loc[df['age'] < 0]

# Delete Specific Values
df2 = df.drop(df[df['age'] < 0].index)
df2.loc[df['age'] < 0]

# Replace with average
df.mean()
df['age'][df['age'] > 0].mean()
df.loc[df['age'] < 0, 'age'] = 40.92
df.loc[df['age'] < 0]

df.head(27)

# Missing Values
df.isnull().sum()

df.loc[pd.isnull(df['age'])]
df['age'].fillna(df['age'].mean(), inplace = True)
df.loc[pd.isnull(df['age'])]

############################
### Training and testing ###
############################

type(df)

# X = variáveis explicativas 
# Y = variável dependente/explicada

X_credit = df.iloc[:, 1:4].values
X_credit

y_credit = df.iloc[:, 4].values
y_credit