import mnist_loader
import network
import pickle

# Definir las salidas
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

# Convertir en listas
training_data = list(training_data)
test_data = list(test_data)



