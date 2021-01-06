import pandas as pd

#Faz o pre processamento dos dados
base = pd.read_csv('credit_data.csv')
#Corrige o valor da idade negativa
base.loc[base.age < 0, 'age'] = 40.92
               
#Divide em previsores e a classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Faz a correção dos valores faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

#Escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Divide a base de dados em treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


#Pre processamento da rede neural
from sklearn.neural_network import MLPClassifier
#Faz o treinamento na rede neural
classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.0000010,
                              solver = 'adam',
                              hidden_layer_sizes=(100),
                              activation='relu')
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Mostra o resultado e a matriz com erros e acertos
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)