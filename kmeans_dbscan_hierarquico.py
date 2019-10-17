
#Importa as bibliotecas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np

#Gera 1500 dados aleatorios
x, y = datasets.make_moons(n_samples = 1500, noise = 0.09)
#Printa o grafico de x y
plt.scatter(x[:, 0], x[:, 1], s = 5)

cores = np.array(['red', 'blue'])

#Gera o agrupamento kmeans e mostra o grafico
kmeans = KMeans(n_clusters = 2)
previsoes = kmeans.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])

#Gera o agrupamento hierarquico e mostra o grafico
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
previsoes = hc.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])

#Faz o agrupamento dbscan e mostra o grafico
dbscan = DBSCAN(eps = 0.1)
previsoes = dbscan.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])