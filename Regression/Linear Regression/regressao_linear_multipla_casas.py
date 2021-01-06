import pandas as pd

base = pd.read_csv('house_prices.csv')

x= base.iloc[:, 3:19].values
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
#Score da relação enrte x e y
score = regressor.score(x_treinamento, y_treinamento)

#Faz o treinamento
previsoes = regressor.predict(x_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error
#Mostra a diferença entre o valor real e o valor treinado
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

#Mostra o score da base teste
regressor.score(x_teste, y_teste)

#Mostra o B0
regressor.intercept_
#mOSTRA OS B1, que no caso sao varios , pois é uma regresssao multipla
regressor.coef_
len(regressor.coef_)