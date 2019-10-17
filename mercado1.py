import pandas as pd

#Faz a leitura da base de dados e armazena em uma variavel
#header = None diz que nao tem cabeçario, logo a primeira linha ja conta
dados = pd.read_csv('mercado.csv', header = None)
transacoes = []

#Coloca os valores em uma matriz transaçao
#Pois o sistema apriori usa o sistema de matrizes
for i in range(0, 10):
    transacoes.append([str(dados.values[i,j]) for j in range(0, 4)])

#Importa a biblioteca apriori
from apyori import apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)

resultados = list(regras)
resultados

resultados2 = [list(x) for x in resultados]
resultados2
resultadoFormatado = []
for j in range(0, 5):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado
