import pandas as pd

base = pd.read_csv('plano_saude2.csv')

#Divide a base de dados em x e y
X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

#Escalonamento dos dados
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#Importa a biblioteca de redes neurais
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
#Treinamento com redes neurais
regressor.fit(X, y)
#score dessa base de dados
regressor.score(X, y)

#Grafico com redes neurais
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

#Faz uma previsao, com inversao do escalonamento
#Continua dando um erro de array
previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(40)))