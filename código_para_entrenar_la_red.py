import mnist_loader
import network
import pickle

# Definir las salidas
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

# Convertir en listas
training_data = list(training_data)
test_data = list(test_data)

net = network.Network([784,30,10])
"""
Crea una instancia de la clase Network con la arquitectura:

784 neuronas en la capa de entrada,

30 neuronas en la capa oculta,

10 neuronas en la capa de salida.
"""

