
#Faz a importação das bibliotecas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#Le a base de dados
# header = 1 informa que so vai pegar a partir da segunda linha
base = pd.read_csv('credit_card_clients.csv', header = 1)

#Faz o somatorio das dividas em geral e adiciona no final da base
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

#Cria uma nova base de daods somente com os valores da coluna 1,2,3,4,5,25
X = base.iloc[:,[1,2,3,4,5,25]].values
#Faz o escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

#O objetivo é escolher o numero de cllusters de acordo com a ultima diferença grande de wcss 
#Faz o calculo do valor do wss para cada numero de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    #Faz o treinamento de acordo com o n de clusters
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#Printa o grafico de wcss por numero de clusters
plt.plot(range(1, 11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')


#Começa o programa oficial com o numero de clusters encoontrado ideal
kmeans = KMeans(n_clusters = 4, random_state = 0)
#Faz a aprendizagem e ja indica a qual cluster ele pertence
previsoes = kmeans.fit_predict(X)

#Mostra a lista de clientes com seus respectivos clusters
lista_clientes = np.column_stack((base, previsoes))
#Ordena a lista
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]