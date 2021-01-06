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
#Faz a geração de regras para a matriz transação
#min_support = 0.3 é o suporte que define o calculo inicial
#min_confidence = 0.8 é a confiança minima
#min_lift = 2 é o numero minimo do lift
#min_length = 2 é o numero minimo de atrubutos da sua regra
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)

#Mostra as regras em lista
resultados = list(regras)
resultados

#Cria uma segunda variavel resultado com uma melhor organizada, com uma separação por linha
resultados2 = [list(x) for x in resultados]
resultados2

#Mostra as regras de forma ordenada, com o parametro range informando quantos voce quer
resultadoFormatado = []
for j in range(0, 3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])
resultadoFormatado
