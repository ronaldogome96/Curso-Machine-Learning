import pandas as pd

base = pd.read_csv('house_prices.csv')

#Divide a base em x e y
x = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

#Divide em base de teste e de treinamento
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

#Faz o treinamento por meio de arvores de decisao
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_treinamento, y_treinamento)
score = regressor.score(x_treinamento, y_treinamento)

#Faz as previsoes de acorodo com a arvore de decisao
previsoes = regressor.predict(x_teste)

#Mostra a diferença entre os teste e previsoes
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

#Mostra o score das bases de teste
regressor.score(x_teste, y_teste)

#Provaçvelmente se adptou muito a base de teste
