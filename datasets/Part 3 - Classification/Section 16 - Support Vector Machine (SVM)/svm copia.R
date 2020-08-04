# SVM

# Importar el dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, c("Gender", "Age", "Purchased")] #Aquí me quedo con las variables que me intersan
                                                    #por nombre de columna

# dataset$Gender = factor (dataset$Gender,
#                          levels = c("Male", "Female"),
#                          labels = c(10, 25))

#install.packages("dummies")

library(dummies)

dataset.dummy <- dummy.data.frame(dataset, sep =".")

#dataset$Gender <- as.numeric(as.character(dataset$Gender)) #Transformo las etiquetas del factor a números

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset.dummy$Purchased, SplitRatio = 0.75)
training_set = subset(dataset.dummy, split == TRUE)
testing_set = subset(dataset.dummy, split == FALSE)

# Escalado de valores
training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

# Ajustar el SVM con el conjunto de entrenamiento.
#install.packages("e1071")
library(e1071)
classifier = svm(formula = Purchased ~., 
                 data = training_set,
                 type = "C-classification",
                 kernel = "radial")

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set [,-4])

# Crear la matriz de confusión
cm = table(testing_set[, 4], y_pred)
cm

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
X3 = seq(min(set[, 3]) - 1, max(set[, 3]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2, X3)
colnames(grid_set) = c('Gender.Female', 'Gender.Male', 'Age')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'Género', ylab = 'Edad',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 4] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

