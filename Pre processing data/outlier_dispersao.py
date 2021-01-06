import pandas as pd

#Le a base de dados 
base = pd.read_csv('credit_data.csv')
#Tira as linhas com valores faltantes
base = base.dropna()
#Substitui valores negativos pela media
base.loc[base.age < 0, 'age'] = 40.92


# Mostra o grafico income x age
import matplotlib.pyplot as plt
plt.scatter(base.iloc[:,1], base.iloc[:,2])

#Mostra o grafico income x loan
plt.scatter(base.iloc[:,1], base.iloc[:,3])

#Mostra o grafico age x loan
plt.scatter(base.iloc[:,2], base.iloc[:,3])

base_census = pd.read_csv('census.csv')

#Mostra o grafico age x final weight
plt.scatter(base_census.iloc[:, 0], base_census.iloc[:,2])