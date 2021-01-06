import pandas as pd

base = pd.read_csv('plano_saude2.csv')

#Faz a divisao da base de dados
x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

#Faz a  Regressão linear simples
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x, y)
#Mostra o score da linear simples
score1 = regressor1.score(x, y)

#Faz previsao de um valor aleatorio
#Nao da certo no meu pc
regressor1.predict(40)

#Printa o grafico de acordo com a base de dados
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor1.predict(x), color = 'red')
plt.title('Regressão linear')
plt.xlabel('Idade')
plt.ylabel('Custo')

#Faz a Regressão polinomial
#Faz o pre processamento para caracteristicas polonomiais
from sklearn.preprocessing import PolynomialFeatures
#(degree = 4) informa o valor da potencia
poly = PolynomialFeatures(degree = 4)
#Faz a trasnformção de acordo com a caracteristica polinomial
x_poly = poly.fit_transform(x)

#Faz o treinamento da base de dados com valores polinomiais
regressor2 = LinearRegression()
regressor2.fit(x_poly, y)
score2 = regressor2.score(x_poly, y)

#Nao roda no meu pc, erro de array
regressor2.predict(poly.transform(40))

#Printa o grafico da polinomial
plt.scatter(x, y)
plt.plot(x, regressor2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')