# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:57:59 2019

@author: ronal
"""

import pandas as pd
base=pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#Nesta parte é feita a troca de strings para valores numericos, pois a maquina nao aprende com strings
#e sim com numeros discretos
#é feita a transformação em colunas com atributos strings
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,1] = labels = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labels = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labels = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labels = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labels = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labels = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labels = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labels = labelencoder_previsores.fit_transform(previsores[:,13])

    