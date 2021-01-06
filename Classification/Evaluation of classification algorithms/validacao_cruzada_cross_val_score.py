
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

#Função especifica para fazer a validação cruzada
from sklearn.model_selection import cross_val_score
#Usa o naive bayes pois gasta menos tempo de processamento
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
#Faz a validação cruzada entre o classificador, previsores e a classe
#CV =10 informa que sera feito 10 testes
resultados = cross_val_score(classificador, previsores, classe, cv=10)
#Faz a media dos 10 resultados
resultados.mean()
#Mostra o desvio padrao
resultados.std()

