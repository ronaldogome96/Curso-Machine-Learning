
import pandas as pd
base= pd.read_csv('plano_saude.csv')

#Faz a criação do atributo de X e de Y para a base de plano de saude
x= base.iloc[:, 0].values
y= base.iloc[:, 1].values

import numpy as np
#Cria uma matriz que mostra a correlaçao entre x e y
#Essa corralação varia entre -1 e 1
#Quanto mais proxima de 1 e -1, mais forte é a correlação
correlacao = np.corrcoef(x,y)

#Tranforma em formato de matriz
x = x.reshape(-1,1)

#Importa a bilioteca de regressao linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Faz o treinamento da base de dados
regressor.fit(x,y)

#b1
regressor.intercept_

#b0
regressor.coef_

import matplotlib.pyplot as plt
#Mostra o grafico
plt.scatter(x,y)
#Faz a previsao dos valores e cria a reta no grafico
plt.plot(x, regressor.predict(x), color= 'red')
plt.title('Regressao Linear simples')
plt.xlabel('Idade')
plt.ylabel('Custo')

#Esse previsao 1 nao deu certo no meu pc, deu um erro de array
previsao1 = regressor.predict(40)
#Ja esse deu certo, criou normalmente
previsao2 = regressor.intercept_ + regressor.coef_ * 40

#Mostra o quanto ele esta adptado ao data set
score = regressor.score(x,y)

from yellowbrick.regressor import ResidualsPlot
#Mostra os erros e a distancia de cada ponto para a reta
visualizador = ResidualsPlot(regressor)
visualizador.fit(x,y)
visualizador.poof()
