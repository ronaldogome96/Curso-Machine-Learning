
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

#Cria 2 neuronios na camada de entrada
#Cria tres neuronios na camada oculta
#1 neuronio na camada de saida
#outclass = SoftmaxLayer cria ou edita as funções
#pode editar de qualquer uma
rede = buildNetwork(2, 3, 1)

#Mostra as funções das camadas
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])

#Base de dados
#2 previsores e 1 classe
base = SupervisedDataSet(2,1)
#Cria base de dados XOR
base.addSample((0,0),(0,))
base.addSample((0,1),(1,))
base.addSample((1,0),(1,))
base.addSample((1,1),(0,))
print(base['input'])
print(base['target'])

#Faz o treinamento, passando os parametros 
treinamento = BackpropTrainer(rede, dataset=base, learningrate = 0.01, momentum= 0.06)

for i in range(1, 30000):
    #Faz os ajustes dos pesos
    erro= treinamento.train()
    if (i%1000) == 0:
        print("erro: %s" %erro)

print(rede.activate([0,0]))
print(rede.activate([1,0]))
print(rede.activate([0,1]))
print(rede.activate([1,1]))    
    
    