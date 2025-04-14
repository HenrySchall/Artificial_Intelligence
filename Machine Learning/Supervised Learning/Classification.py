import string as string
import random as random
import radian as rd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

####################
### Introduction ###
####################

# Taken from: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

data = "https://drive.google.com/uc?export=download&id=1wMapByTvMFt16zz9Bd2643eTHJXtEhnX"
df = pd.read_csv(data)

Traipubtos diferentes considera que nao ha correlacao entre as variaveis explicwtivaw 

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

# LabelEncoder -> tranformar variaveis categorias em númericas sequências (1,2,3,4...)
# OnehotEncoder -> tranforma em variáveis dummy (1 ou 0)

#############################
### Escalation Management ###
#############################

type(df)

# X = variáveis explicativas 
# Y = variável dependente/explicada

X_credit = df.iloc[:, 1:4].values
X_credit

y_credit = df.iloc[:, 4].values
y_credit

# Standardisation Data (leaving on the same scale)
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

# Normalization Data
scaler_credit = MinMaxScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit[:,0].min()
X_credit[:,1].min()
X_credit[:,2].min()

X_credit[:,0].max()
X_credit[:,1].max()
X_credit[:,2].max()

############################
### Training and testing ###
############################

X_credit.shape
y_credit.shape

X_credit_treinamento, X_credit_teste = train_test_split(X_credit, test_size = 0.25, random_state = 0)

y_credit_treinamento, y_credit_teste = train_test_split(y_credit, test_size = 0.25, random_state = 0)

# Records for Training
X_credit_treinamento.shape #(1500 = number of records, "3" = three variables dependent)
y_credit_treinamento.shape #(1500 = number of records, "" = one variables dependent)

# Records for Test
X_credit_teste.shape
y_credit_teste.shape

# Salvar Pickle
import pickle

with open('credit.pkl', mode = 'wb') as f:
  pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)

###################
### Naïve Bayes ###
###################

df2 = pd.read_excel("https://drive.google.com/uc?export=download&id=1a1vV-zrje_wgN9ucN8yJ2DF_N0jGuWAI")
df2

#####################
### LabelEncoder ###
#####################

X_risco_credito_label = df2.iloc[:, 0:4].values
X_risco_credito_label

y_risco_credito_label = df2.iloc[:, 2].values
y_risco_credito_label

label_encoder_history = LabelEncoder()
label_encoder_debit = LabelEncoder()
label_encoder_guarantee= LabelEncoder()
label_encoder_income = LabelEncoder()

X_risco_credito_label[:,0] = label_encoder_history.fit_transform(X_risco_credito_label[:,0])
X_risco_credito_label[:,1] = label_encoder_debit.fit_transform(X_risco_credito_label[:,1])
X_risco_credito_label[:,2] = label_encoder_guarantee.fit_transform(X_risco_credito_label[:,2])
X_risco_credito_label[:,3] = label_encoder_income.fit_transform(X_risco_credito_label[:,3])

X_risco_credito_label

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito_label, y_risco_credito_label)

# Modelo 1 -> história boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2)
# Modelo 2 -> história ruim (2), dívida alta (0), garantias adequada (0), renda < 15 (0)

forecast = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
forecast 

#####################
### OneHotEncoder ###
#####################

df3 = df2
df3['debit'] = df3['debit'].replace({'alta': 1, 'baixa': 0})
df3 

X_risco_credito_onehot = df2.iloc[:, 3:7].values
X_risco_credito_onehot

y_risco_credito_onehot = df2.iloc[:, 3].values
y_risco_credito_onehot

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito_onehot, y_risco_credito_onehot)

# Modelo 1 -> história boa (0), dívida alta (0), garantias nenhuma (1), renda > 35 (2)
# Modelo 2 -> história ruim (2), dívida alta (0), garantias adequada (0), renda < 15 (0)

#forecast = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
#forecast 

########################
### Pratical Example ###
########################

# Taken from: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

import pickle
with open('/content/census.pkl', 'rb') as f:
  X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape

X_credit_teste.shape, y_credit_teste.shape

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

forecast = naive_credit_data.predict(X_credit_teste)
forecast 