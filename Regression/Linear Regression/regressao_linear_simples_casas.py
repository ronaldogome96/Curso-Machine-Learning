import pandas as pd

base = pd.read_csv('house_prices.csv')

x= base.iloc[:, 5:6].values
y= base.iloc[:, 2].values

#Divide a base de dados em treinamento e teste
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,
                                                                  test_size= 0.3,
                                                                  random_state=0 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Faz o treinamento com regressao linear
regressor.fit(x_treinamento, y_treinamento)
#Mostra o score de treinamento
score= regressor.score(x_treinamento, y_treinamento)

import matplotlib.pyplot as plt
#Mostra o grafico do treinamento
plt.scatter(x_treinamento, y_treinamento)
#Mostra a linha no grafico
plt.plot(x_treinamento, regressor.predict(x_treinamento), color='red')

#Mostra as previsoes de acordo com o treinamento
previsoes= regressor.predict(x_teste)

#Mostra a diferença dos resultados obtidos para o real
resultado = abs(y_teste-previsoes)
#Mostra a media do erro de previsao
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)


import matplotlib.pyplot as plt
#Mostra o grafico do teste
plt.scatter(x_teste, y_teste)
#Mostra a linha no grafico
plt.plot(x_teste, regressor.predict(x_teste), color='red')

#Mostra o score do teste
regressor.score(x_teste, y_teste)
