import pandas as pd

#Faz a leitura do arquivo que contem a base de dados
base= pd.read_csv('risco_credito2.csv')

#Cria as classes separadas
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values  

##Faz as substituições de valores nominais para valores discretos
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:,0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:,1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:,3])

#Fez a importação da biblioteca que faz aregressao liner
from sklearn.linear_model import LogisticRegression
#Faz a regressao
classificador = LogisticRegression()
classificador.fit(previsores, classe)
#Printa o valor de b0
print(classificador.intercept_)
#Printa os valores de B, os parametros
print(classificador.coef_)

#Exemplo: Historica boa, divida alta, garantia nenhuma, renda>35
#Exemplo: Historico ruim, divida alta, garantia adequada, renda<15
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
#Predict faz as contas de probabilidade de chance do resultado
print(resultado)

#Mostra a probabilidade, de acorodo com o grafico gerado, 0 alto e 1 baixo
resultado2 = classificador.predict_proba([[0,0,1,2], [3,0,0,0]])
print(resultado2)

