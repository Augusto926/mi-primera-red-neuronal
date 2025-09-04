import mnist_loader
import network
import pickle

# Define la salida de mnist_loader.load_data_wrapper()
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

# Convertir en listas
training_data = list(training_data)
test_data = list(test_data)

# Crea la arquitectura de la red
net = network.Network([784,30,10])
"""
Crea una instancia de la clase Network con la arquitectura:

784 neuronas en la capa de entrada,

30 neuronas en la capa oculta,

10 neuronas en la capa de salida.
"""

# Define los hiperparamétros del metodo "GSD" de la clase "network"
net.SGD(training_data, 30, 10, 0.1, test_data=test_data)
"""
El valor de cada hiperparametro es:

epochs=30,

mini_batch_size=10,

eta=0.1.
"""

# Crea un fichero llamado "mejora_2_de_la_red_network.pkl" en modo escritura binaria 'wb'
archivo = open("mejora_2_de_la_red_network.pkl",'wb')
# Guarda el objeto net y lo escribe en el fichero
pickle.dump(net, archivo)
# Cierra el fichero y asegura que los datos se hayan escrito
archivo.close()


