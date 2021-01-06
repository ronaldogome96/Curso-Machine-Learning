import pandas as pd

base = pd.read_csv('house_prices.csv')

#Divide a base de dados em x e y
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

#Faz o escalonamento, pois usaremos o rbf que tem resultado melhores
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#Divide em base de testes e treinamento
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

#Faz a importação do svr
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#Faz o treinamento
regressor.fit(X_treinamento, y_treinamento)
#Score do treinamento, a correlaçao
score = regressor.score(X_treinamento, y_treinamento)

#Score do teste
regressor.score(X_teste, y_teste)

#Faz os previsores
previsoes = regressor.predict(X_teste)
#Retorna do escalonamento
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

#Mostra a diferença entre o teste e as previsoes
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)



