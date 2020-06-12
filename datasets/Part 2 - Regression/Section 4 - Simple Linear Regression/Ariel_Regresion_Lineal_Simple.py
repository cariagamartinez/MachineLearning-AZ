#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:33:07 2020

@author: macbookariel
"""
# Regresión lineal simple: el primer paso será copiar el documento de data_preprocessing.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("Salary_Data.csv") 

#En este caso, al ver la variable, tenemos que darle a format %,1 para ver 1 posicición decimal.
# Recordemos que la columna llamada "Index en el variable explorer NO ES UNA COLUMNA Y QUE LAS COLUMNAS EMPIEZAN
# A NOMBRARSE EN EL CERO.

X = dataset.iloc[:, :-1].values #Variable independiente = años de experiencia. Ubicada en la anteúltima posición (-1)
y = dataset.iloc[:, 1].values # Variable dependiente = a predecir = salario. Ubicada en la columna 1


# Dividir el data set en conjunto de entrenamiento y en conjunto de testing.
# En este caso vamos a tomar 10 para testing (1/3) y el resto para entrenamiento (20).

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)

# Escalado de variables. En el caso de la regresión lineal, el modelo no requiere escalado.

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Crear modelo de regresión lineal simple con el conjunto de entrenamiento.

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train) #La máquina ha aprendido, entonces, con las variables que le suministramos (igual tamaño en ambas variables!!)


# Predecir el conjunto de test. Para ello crearemos un vector de datos con los datos de predicción para obtener la variable 
# dependiente que nos devuelve el modelo. Observemos que la variable a ser suministrada solo es la independiente (X_test) y
# el modelo hace la predicción y la guarda en y_pred. Es decir, usando la X_test (años de experiencia) quiero que prediga el sueldo
# y lo guarde en y_pred

y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento. Vamos a generar un scatter plot (nube de dispersión). Vamos a usar pyplot.

plt.scatter(X_train, y_train, color = "red")

# Vamos a hacer un scatter plot donde la X es el grupo de entrenamiento y la y es la predición pero sobre X_train,
# así vemos las dos variables

plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Vamos a hacer un scatter plot para ver cómo quedan los datos de test y cómo se ajusta a ellos la recta de regresión

plt.scatter(X_test, y_test, color = "red")

plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de experiencia (Conjunto de testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()






















