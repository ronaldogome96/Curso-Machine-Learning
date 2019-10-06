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

