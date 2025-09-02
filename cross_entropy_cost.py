import numpy as np
# Esta clase agrupa la lógica de la función de costo de Entropía Cruzada (cross-entropy).
class CrossEntropyCost(object): 

    @staticmethod 
    def fn(a, y):
        """
        Retorna el costo asociado a la salida "a" y la salida deseada "y". Nota que np.nan_to_num 
        se usa para tener estabilidad númerrica. Si "a" o "y" tienen 1.0 en las mismas entradas, 
        la expresión (1-y)*np.log(1-a) puede producir nan por log(0), y np.nan_to_num corrige esos nan a 0.0. 
        np.sum(...) suma todos los elementos resultantes y devuelve el costo escalar total. 
        """    
        return np.sum(-y*np.log(a)-(1-y)*np.log(1-a))
    
    def delta(a, y):
        """
        Retorna la señal de error en la capa de salida 
        (el vector δ = ∂C/∂z para la capa de salida).
        """
        return (a-y)


