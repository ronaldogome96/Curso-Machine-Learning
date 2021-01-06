import Orange

base = Orange.data.Table('credit_data.csv')
#Mostra as informações dos dados carregados
base.domain

#Faz a divisao da base, em 25% pra treinamento da maquina e 75% pra teste
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
len(base_treinamento)
len(base_teste)


#Gera as regras
cn2_learner = Orange.classification.rules.CN2Learner()
#Faz o treinamento da base de acorodo com as regras
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)
    
#Essa parte da o erro Classification metrics can't handle a mix of binary and continuous targets
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))