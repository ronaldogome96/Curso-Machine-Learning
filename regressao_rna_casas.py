import pandas as pd

base = pd.read_csv('house_prices.csv')

#Divide em x e y
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

#Faz o escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#Divisao entre treinamento e teste
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

#Faz a importação da biblioteca da rede neural
from sklearn.neural_network import MLPRegressor
#hidden_layer_sizes = (9,9) numero de camadas ocultas e neuronios nelas
regressor = MLPRegressor(hidden_layer_sizes = (9,9))
#Treinamento
regressor.fit(X_treinamento, y_treinamento)
#Score do treinamento
score = regressor.score(X_treinamento, y_treinamento)
#Score da base teste
regressor.score(X_teste, y_teste)

#Previsoes a partir do treinamento
previsoes = regressor.predict(X_teste)
#Voltando do escalonamento
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

#Mostra a diferença entre os dados finais
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_teste, previsoes)



