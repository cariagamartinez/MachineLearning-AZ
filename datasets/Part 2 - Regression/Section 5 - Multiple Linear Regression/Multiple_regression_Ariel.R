# Regresión lineal múltiple

# Como siempre, lo primero es copiar toda la plantilla de pre procesado de datos y cambiar
# lo que sea necesario

dataset = read.csv('50_Startups.csv')

# Codificar las variables categóricas
dataset$State = factor(dataset$State,
                         levels = c("California", "Florida", "New York"),
                         labels = c(1, 2, 3))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo con el training set, usando la función lm. La fórmula es la
# columna a predecir en función (~) de tod las otras (.)
#

regression = lm(formula = Profit ~ .,
                data = training_set)
# Ahora consultamos los datos con la función summary

summary(regression)

# Predecir los resultados con el conjunto de testing

y_pred = predict(regression, newdata = testing_set)

# Construir un modelo óptimo con la Eliminación hacia atrás. Recordemos que lo anterior
# era generar el modelo con el set de entrenamiento y ver cómo predecía el set de testing.
# Ahora vamos a construir el modelo usando todos los valores (data = dataset) e
# indicando directamente todos los nombres de variables.
# Empezamos con todas las variables e iremos eliminando columna a columna

SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regression)

# Luego de ver el summary, vamos a eliminar la variable state

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

# Luego de ver el summary, vamos a eliminar la variable administration

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

# Luego de ver el summary, vamos a eliminar la variable marketing

regression = lm(formula = Profit ~ R.D.Spend,
                data = dataset)
summary(regression)

# Tendrímos la posibilidad de hacerlo automático con lo siguiente


backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

















