
library(tfruns)

# 1 hyperparametro: numero de capas ocultas
# 1 " : nodos hidden layers (opcional)
# 1 dropout (0.4 o 0)
# 1 layer_dense(kernel_regularizer = NULL / L2)
# En total habrá 8 ejecuciones diferentes. Crear una matriz y que por cada ejecucion lea una fila.
# El fit final debe tener un validation_split = 0 y un callback que guarde los pesos.


training_run('Practica_dnn_2.R',
             flags = c(drop = 0.4, regul = 0.001, l3 = 0))

training_run('Practica_dnn_2.R')
