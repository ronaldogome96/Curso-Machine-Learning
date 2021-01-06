import pandas as pd

#Le o arquivo 
base = pd.read_csv('credit_data.csv')
#Exclui os valores faltantes
base = base.dropna()

#Importa a biblioteca
from pyod.models.knn import KNN
#Faz a detecção de outlier por esse modelo
detector = KNN()
detector.fit(base.iloc[:,1:4])

#Mostra se o dado é outlier ou nao
previsoes = detector.labels_
#Mostra a confiança desse dado
confianca_previsoes = detector.decision_scores_

#Faz uma lista com as linhas que contem outliers
outliers = []
for i in range(len(previsoes)):
    #print(previsoes[i])
    if previsoes[i] == 1:
        outliers.append(i)

#Faz uma lista com os outliers
lista_outliers = base.iloc[outliers, :]
    