import Orange

#Faz a importação dessa função que ira fazer o  carregaomento dos dados
base = Orange.data.Table('risco_credito.csv')
#Mostra as informações dos dados que foram carregados
base.domain

#Faz a indução de regras, gera as regras
cn2_learner = Orange.classification.rules.CN2Learner()
#cria o classificador das regras
classificador = cn2_learner(base)

#Imprime as regras que foram geradas para ese arquivo
for regras in classificador.rule_list:
    print(regras)
    
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
for i in resultado:
    #Mostra os valores no formato do arquivo
    print(base.domain.class_var.values[i])
    