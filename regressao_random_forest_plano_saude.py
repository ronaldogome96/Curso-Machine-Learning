import pandas as pd

base = pd.read_csv('plano_saude2.csv')

#Separa em x e y
x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

#Importa a biblioteca do random forest
from sklearn.ensemble import RandomForestRegressor
#n_estimators = 10 sao o numero de arvores que voce quer usar
regressor = RandomForestRegressor(n_estimators = 5)
#Treinamento
regressor.fit(x, y)
score = regressor.score(x, y)

#Faz a regressao polinomial
import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)

#Grafico de acordo com o random forest
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
plt.title('Regressão com random forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao = regressor.predict(40)