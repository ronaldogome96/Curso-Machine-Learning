# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:04:53 2019

@author: ronal
"""
import pandas as pd

#Faz a leitura do arquivo que contem a base de dados
base= pd.read_csv('risco_credito.csv')

#Cria as classes separadas
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values  

##Faz as substituições de valores nominais para valores discretos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:,1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])

# Aplica a naive de bayes, para fazer a tabela de aprendizagem
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
#Fit é o treinamento do algoritmo, gerando a tabela de probabilidade
classificador.fit(previsores, classe)

#Exemplo: Historica boa, divida alta, garantia nenhuma, renda>35
#Exemplo: Historico ruim, divida alta, garantia adequada, renda<15
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
#Predict faz as contas de probabilidade de chance do resultado


#Printa os tipos de resultados
print(classificador.classes_)
#Mostra a quantidade de cada resultado na tabela original
print(classificador.class_count_)
#Probabilidades a priori
print(classificador.class_prior_)





