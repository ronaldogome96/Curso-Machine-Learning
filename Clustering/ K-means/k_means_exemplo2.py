import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#samples_generator é o gerador automatico de registros
from sklearn.datasets.samples_generator import make_blobs

#n_samples = 200 numero de registros que quero gerar
# centers = 5 numero de centroides
x, y = make_blobs(n_samples = 200, centers = 5)
#Printa o grafico de dados
plt.scatter(x[:,0], x[:,1])


kmeans = KMeans(n_clusters = 5)
#Aprendizagem com kmeans
kmeans.fit(x)

#Mostra o resultado das previsoes do aprendizagem 
previsoes = kmeans.predict(x)

#c = previsoes coloca a cor pra cada classe diferente
plt.scatter(x[:,0], x[:,1], c = previsoes)
