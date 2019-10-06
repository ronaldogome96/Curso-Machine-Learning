# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:09:58 2019

@author: ronal
"""

import pandas as pd
base = pd.read_csv('credit_data.csv')
#descreve os dados
base.describe()
#localiza algum valor na base de dados
base.loc[base['age']<0]
# apagar a coluna
base.drop('age', 1 , inplace=True)
# Apagar somenete os registros com problemas
base.drop(base[base.age<0].index, inplace=True)
#preeencher os valores manualmente
# preencher esses valores com a media
base.mean()
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age<0 , 'age'] =40.92

#Cria duas novas base de dados, com valores especificos
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values

#Faz a substituição de valores faltantes para a media
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer= imputer.fit(previsores[:, 0:3])
previsores[: , 0:3] = imputer.transform(previsores[:,0:3])

#Faz uma padronização dos dados, como se fosse uma transformação linear
#Transforma mm dados mais juntos, com uma diferença menor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores=scaler.fit_transform(previsores)






