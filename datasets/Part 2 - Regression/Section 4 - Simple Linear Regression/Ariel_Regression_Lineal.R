# Regresión lineal simple

# Plantilla para el Pre Procesado de Datos
# Importar el dataset
dataset = read.csv('Salary_Data.csv')

# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) #qué tamaño de la muestra se quedará en entrenamiento
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento. Vamos a usar la función lm. 
# Vamos a guardar la función lm en la variable regressor. Esta función necesita la fórmula donde indicamos
# la variable X (Salary) en función de (~) la variable y (YearsExperiencie).
# Notar que los nombres de X y de Y son los que aparecen en las columnas del dataset.
# Luego indicamos el conjunto de datos, que en este caso es el training set.

regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Ahora vamos a hacer un resumen estadístico de la variable 'regressor' que guarda la regresión lineal.

summary(regressor)


# Predecir resultados con el conjunto de test (testing). Para este paso es necesario que las columnas se llamen igual!!!
# Vamos a crear un vector de predicción (y_pred) y vamos a usar la función predict que requiere el modelo
# (guardado en regressor) y los nuevos datos (newdata) a predecir (testing_set)

y_pred = predict(regressor, newdata = testing_set)

# Visualizar los datos. Vamos a cargar la librería ggplot2

library(ggplot2)

ggplot() +
  geom_point(aes(x= training_set$YearsExperience, 
                 y = training_set$Salary), colour = "red")+
  geom_line(aes(x= training_set$YearsExperience, 
                 y = predict(regressor, newdata = training_set)),
                colour = "blue")+ 
  ggtitle("Sueldo vs Años de experiencia (conjunto de entrenamiento)") +
  xlab("Años de experiencia") +
  ylab("Sueldo en dólares")

#Visualización de los resultados en el conjunto de testing

ggplot() +
  geom_point(aes(x= testing_set$YearsExperience, 
                 y = testing_set$Salary), colour = "red")+
  geom_line(aes(x= training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = "blue")+ 
  ggtitle("Sueldo vs Años de experiencia (conjunto de testing)") +
  xlab("Años de experiencia") +
  ylab("Sueldo en dólares")



























