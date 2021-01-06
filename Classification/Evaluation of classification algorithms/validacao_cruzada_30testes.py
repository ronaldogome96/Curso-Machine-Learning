
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Ajuda na escolhe os valores mais imoprtantes
import numpy as np


#Garante a melhor distribuição entre as classes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
resultados30= []
#Faz  validacao 30 vezes
for i in range(30):

    #Usa o K =10
    #random_state = i garante que sera feito as 30 vezes, cada uma com 10 interações
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    #Resultados de uma rodada
    resultados1 = []
    #Mostra os indices com a validação cruzada
    for indice_treinamento, indice_teste, in kfold.split(previsores,
                                                     np.zeros(shape=(previsores.shape[0], 1))):
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = LogisticRegression()
        #classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0)
        #classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        #classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation = 'relu',
                              batch_size=200, learning_rate_init=0.001)
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    #Converte o resultado parcial das 10 interações
    resultados1= np.asarray(resultados1)
    media = resultados1.mean()
    #Adiona a media desses 10 ao resultado30
    resultados30.append(media)
#Convete o resultado 30
resultados30 = np.asarray(resultados30)
#Mosrta os valores das medias para alimentar a tabela
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.',','))
    
