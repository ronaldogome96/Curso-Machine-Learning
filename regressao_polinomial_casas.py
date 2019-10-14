import pandas as pd

#Faz a importação dos dados
base = pd.read_csv('house_prices.csv')

#Divide a base de dados
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

#Divide em base de teste e de treinamento
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

#Faz o pre processamento polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
#Faz a trasnformação polinomial
X_treinamento_poly = poly.fit_transform(X_treinamento)
X_teste_poly = poly.transform(X_teste)


from sklearn.linear_model import LinearRegression
#Faz a regressao com os dados ja em polynomial
regressor = LinearRegression()
regressor.fit(X_treinamento_poly, y_treinamento)
#Mostra o score obtido do treinamento
score = regressor.score(X_treinamento_poly, y_treinamento)

#Faz a aprendizagem
previsoes = regressor.predict(X_teste_poly)

#Mostra a diferença entre a classe previsores e teste
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)