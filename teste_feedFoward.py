from pybrain.structure import FeedFowardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedFowardNetwork()

CamadaEntrada = LinearLayer(2)
CamadaOculta = SigmoidLayer(3)
CamadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(CamadaEntrada)
rede.addModule(CamadaOculta)
rede.addmodule(CamadaSaida)
rede.addModule(bias1)
rede.aadModule(bias2)

entradaOculta = FullConnection(CamadaEntrada, CamadaOculta)
ocultaSaida = FullConnection(CamadaOculta, CamadaSaida)
biasOculta = FullConnection(bias1, CamadaOculta)
biasSaida = FullConnection(bias2, CamadaSaida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(biasOculta.params)
print(biasSaida.params)

