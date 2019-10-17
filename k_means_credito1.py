
#Faz a importação das bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

#Le a base de dados
# header = 1 informa que so vai pegar a partir da segunda linha
base = pd.read_csv('credit_card_clients.csv', header = 1)

#Faz o somatorio das dividas em geral e adiciona no final da base
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

#Cria uma nova base de daods somente com os valores da coluna 1 e 25
X = base.iloc[:,[1,25]].values

#Faz o escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)


wcss = []
#O objetivo é escolher o numero de cllusters de acordo com a ultima diferença grande de wcss 
#Faz o calculo do valor do wss para cada numero de clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    #Faz o treinamento de acordo com o n de clusters
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#Printa o grafico de wcss por numero de clusters
plt.plot(range(1, 11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
