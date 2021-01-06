import pandas as pd

base = pd.read_csv('house_prices.csv')

#Separa em x e y
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

#Divide em  treinamento e teste
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

#Faz a importação do random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
#Aprendizagem
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

#Faz as previsoes, o treinamento da maquina
previsoes = regressor.predict(X_teste)

#Mostra a diferença
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

#Mostra o score da base de teste
regressor.score(X_teste, y_teste)

