import numpy as np
#Importa a biblioteca de visualização de graficos
import matplotlib.pyplot as plt
#Biblioteca de k means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Variavel x é a idade da pessoa
x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
#Y é a renda da pessoa
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  
#Printa o grafico de acordo com x e y
plt.scatter(x,y)

#Dado do tipo array
#Faz uma matriz com os valores relacionados do grafico (x,y)
base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

#Faz o escalonamento da base de dados
scaler = StandardScaler()
base = scaler.fit_transform(base)

#Faz o algoritmo de kmeans
#n_clusters = 3 é o numero de centroides
kmeans = KMeans(n_clusters = 3)
#Aprendizagem da base de dados
kmeans.fit(base)

#Mostra os centroides, com as coordenadas
centroides = kmeans.cluster_centers_
#Mostra a clusters que cada registro pertence
rotulos = kmeans.labels_

#Printa o grafico com cada ponto com a cor correspondente a seu cluster e seu controide com o X
cores = ["g.", "r.", "b."]
for i in range(len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 15)
plt.scatter(centroides[:,0], centroides[:,1], marker = "x")