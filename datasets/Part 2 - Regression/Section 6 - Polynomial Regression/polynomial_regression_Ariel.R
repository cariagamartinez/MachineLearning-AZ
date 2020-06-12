# Regresión Polinómica

#En este ejemplo, vamos a contratar a un empleado y le vamos a preguntar según su categoría
# qué salario cobraba. Luego veremos en nuestra empresa los datos y generaremos un modelo
# que prediga lo que correspondería cobrar en nuestra empresa.

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3] #Recordar que las columnas empiezan en 1 vamos a quedarnos con la 2 y la 3

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)


# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ahora vamos a construir dos modelos: lineal y polinómica

#Ajustar modelo de regresión lineal con el conjunto de datos

lin_reg = lm (formula = Salary ~ .,
              data = dataset) # el punto de la fórmula anterior representa "todas las variables"
                              # En este caso, ver que quiero predecir el salary
                              # en función del resto de variables
summary(lin_reg)

#Ajustar modelo de regresión polinómica.
# Tenemos que construir primero los términos de la regresión polinómica y crearemos las
# variables con los grados adecuados al polinomio

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm (formula = Salary ~ .,
               data = dataset)

summary(poly_reg)

# Visualización del modelo lineal
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

#Visualización del modelo polinómico (genera una recta rígida)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


#Visualización del modelo polinómico "mejorado" (genera una línea más contínua)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# Predicción de nuevos resultados con la regresión lineal: un único número
# En este caso, R necesita un data frame con el dato a predecir, así que creamos un 
# nuevo data frame con un único valor y dentro la columna Level
y_pred = predict (lin_reg, newdata = data.frame (Level = 6.5))


#Predicción de nuevos resultados con la regresión polinómica. Recordemos que ahora 
# necesitamos suministrar todos los datos para la predicción ya que el modelo requiere
# los datos de la columas level2, level3, level4, etc., que en todos los casos es el
# cuadrado, cubo, potencia cuarta, etc, del valor a predecir
y_predi_poly = predict (poly_reg, newdata = data.frame (Level = 6.5,
                                                       Level2 = 6.5^2,
                                                       Level3 = 6.5^3,
                                                       Level4 = 6.5^4))



















