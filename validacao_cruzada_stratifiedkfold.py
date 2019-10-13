
import pandas as pd

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Usa o naive bayes pois gasta menos tempo de processamento
from sklearn.naive_bayes import GaussianNB

#Ajuda na escolhe os valores mais imoprtantes
import numpy as np
a = np.zeros(5)
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))

#Garante a melhor distribuição entre as classes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
#Usa o K =10
#random_state = 0 Faz os daods serem os mesmos, nao importa quantas vezes sejam gerados
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
resultados = []
matrizes = []
#Motrsa os indices com a validação cruzada
for indice_treinamento, indice_teste, in kfold.split(previsores, b):
    #Printa os 10 cruzadas diferentes
    #print('Indice treinamento: ', indice_treinamento, 'Indice teste: ', indice_teste)
    classificador = GaussianNB()
    #Faz a classificação e valida~çao para cada cruzada diferente.
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    #Cria a matriz de confusao para cada validação
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)
#Faz a matriz final com a media do resultado das anteriores
matriz_final = np.mean(matrizes, axis = 0)
#Tranforma a lista em tipo float
resultados = np.asarray(resultados)
resultados.mean()
resultados.std()


