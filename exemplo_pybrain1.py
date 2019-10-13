
#Importa  pacote structure, que faz a estrutura da rede neural
#FeedForwardNetwork tIpo de rede neural
from pybrain.structure import FeedForwardNetwork
#Funções
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
#Faz as conexoes das camadas
from pybrain.structure import FullConnection

#Consegui rodar por um comentario que tem no forum, pois nao estava rodando

#Faz a estrutura da rede
rede = FeedForwardNetwork()
#Faz a criação das camadas, no caso 2 neuronios
camadaEntrada = LinearLayer(2)
#Faz a criação das camadas ocultas, no caso 3 neuronios
camadaOculta = SigmoidLayer(3)
#Faz a criação da camada de saida, no caso com 1 neuronio
camadaSaida = SigmoidLayer(1)

#Faz a criação dos neuronios adicionais
bias1 = BiasUnit()
bias2 = BiasUnit()

#ADICIONANDO AS CAMADAS NA REDE NEURAL
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#Faz as ligações entre as camadas e os bias

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

#Faz a criação final a contrução da rede neural
rede.sortModules()

#Printa a os modulos da rede
print(rede)
#Mostra os pesos aleatorios que foram gerados, da camada de entrada para a oculta
print(entradaOculta.params)
#Mostra os pesos aleatorios que foram gerados, da camada oculta para a de saida
print(ocultaSaida.params)
#Mostra os pesos aleatorios que foram gerados, da camada de bias para a de oculta
print(biasOculta.params)
#Mostra os pesos aleatorios que foram gerados, da camada de bias para a de saida
print(biasSaida.params)




