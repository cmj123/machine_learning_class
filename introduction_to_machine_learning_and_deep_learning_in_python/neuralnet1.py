# Import key libraries
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import SoftmaxLayer

# Build a nnetwork
network = buildNetwork(2,3,1, hiddenclass=TanhLayer, outclass=SoftmaxLayer) # 2 input, 1 hidden layer with 3 neurons and 1 output layer
print(network.activate([1,0]))
print(network['in'])
print(network['hidden0'])
print(network['out'])
