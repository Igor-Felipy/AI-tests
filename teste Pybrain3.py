import scipy
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(2, 1)

ds.addSample((0.8, 0.4), (0.7))
ds.addSample((0.5, 0.7), (0.5))
ds.addSample((1.0, 0.7), (0.95))

nn = buildNetwork(2, 4, 1, bias=True)

trainer = BackpropTrainer(nn, ds)

for i in range(2000):
    print(trainer.train())

while True:
    dormiu = float(input('Dormiu quanto tempo? '))
    estudou = float(input('Estudou quanto tempo? '))
    z = nn.activate((dormiu, estudou))[0] * 10.0
    print(f'Precisão da nota: {str(z)}')
