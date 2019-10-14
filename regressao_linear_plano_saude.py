
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

#b0
regressor.intercept_

#b1
regressor.coef_

