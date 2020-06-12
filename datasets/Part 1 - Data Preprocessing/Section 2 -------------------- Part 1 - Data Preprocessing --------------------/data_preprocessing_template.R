# Plantilla para el Pre Procesado de Datos
# Importar el dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3] # Esto vale para extraer solamente unas columas, en este caso la 2 y 3. 

#Las filas, que son las observaciones, siempre serán usadas por completo, por eso tenemos la coma.

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])