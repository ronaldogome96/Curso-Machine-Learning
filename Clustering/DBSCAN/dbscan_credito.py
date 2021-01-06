
#Importa as bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

#Le a base de dados
# header = 1 informa que so vai pegar a partir da segunda linha
base = pd.read_csv('credit_card_clients.csv', header = 1)
#Faz o somatorio das dividas em geral e adiciona no final da base
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

#Cria uma nova base de daods somente com os valores da coluna 1 e 25
X = base.iloc[:,[1,25]].values
#Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)


#eps = 0.37 distancia maxima do raio para indicar se etsaesta no mesmo grupo
#min_samples = 4 numero minimo de pontos para criar uma regiao
dbscan = DBSCAN(eps = 0.37, min_samples = 4)
previsoes = dbscan.fit_predict(X)
#Cria duas variaveis que vao dizer os valores finais e seus respectivos quantidades
unicos, quantidade = np.unique(previsoes, return_counts = True)

#Printa o grafico com a base original com seus respectivos clusters
plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'orange', label = 'Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]