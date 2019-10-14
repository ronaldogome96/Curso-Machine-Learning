import pandas as pd

base = pd.read_csv('plano_saude2.csv')

#Divide a base em x e y
x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

#Faz a importação da Arvore de decisao de regressao
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
#Faz o treinamento
regressor.fit(x,y)
#Mostra o quanto ele se adaptou
score = regressor.score(x,y)

#Mostra p grafico
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

#Mostra o grafico de acordo com arvores de decisao
import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)
plt.scatter(x, y)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

#Nao roda no meu spyder
regressor.predict(40)