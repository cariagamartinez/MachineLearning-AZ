#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:11:36 2020

@author: macbookariel
"""

#Regresión polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values   #Recordemos que empezamos en cero. 
                                  #La columna cero no interesa en este caso.
y = dataset.iloc[:, 2].values

""" UN COMENTARIO MULTI LÍNEA SE ABRE CON TRES COMILLAS
Y SE CIERRA CON OTRAS TRES!!!

MUY IMPORTANTE: MATRIZ DE DATOS:
    
Realmente solo necesitamos una columna, por lo que podríamos escribir
X = dataset.iloc[:, 1].values, sin embargo, Python la vería como un 
vector, es decir, le faltaría una dimensión (10,).

Si indicamos X = dataset.iloc[:, 1:2].values, cogemos de la primera a la
segunda columna, sin incluir la segunda. Es lo mismo que en el apartado
anterior, pero ahora Python lo ve como matriz, es decir, con dos dimensiones.
(10,1) """


# Dividir el data set en conjunto de entrenamiento y conjunto de testing.
# En este caso tenemos muy pocos datos, así que no dividiremos!!
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables (lo usamos cuando las variables difieren mucho entre sí)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar la regresión lineal con el dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y) #No usamos training y test


# Ajustar la regresión polinómica con todo el dataset.

from sklearn.preprocessing import PolynomialFeatures
"""
Vamos a agregar las potencias polinomiales, es decir, crear una nueva
matriz con todas las potencias"""

poly_reg = PolynomialFeatures(degree = 6)
"""vamos a crear un objeto que guardará a PolynomialFeatures, con grado 2,
es decir cuadrado, luego vamos a crear otro objeto, X_poly, que guardará
el fit transform de poly-reg"""

X_poly = poly_reg.fit_transform(X) 
"""En X_poly ahora tendremos 3 columnas, el término independiente, el
término de grado 1 y el término de grado 2"""

lin_reg_2 = LinearRegression() #Se usa la misma función pero se alteran los datos iniciales
lin_reg_2.fit (X_poly, y)


#Visualización del modelo lineal

plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title ("Modelo de regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel ("Sueldo en $")
plt.show()

# Visualización de los resultados del Modelo Polinómico
""" la idea es que el salto de la línea no sea tan brusco, sino que se 
agregan unos puntos intermedios"""
X_grid = np.arange(min(X), max(X), 0.1) #Para suavizar los trozos de curva
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)

"""veamos qué pasa cuando predecimos con regresión lineal"""
lin_reg.predict([[6.5]])
"""Aquí se pregunta qué cuanto cobraría alguien que estuviera en la posición
6,5, es decir, entre Partner y Senior Partner.
Si ejecuto, veremos que daría un salario de 330K"""

"""veamos qué pasa con la predicción del modelo polinomial"""

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

""" en esta línea, lin_reg_2 va a hacer una predicción (usa el .predict), pero
no puede aceptar un número sin más, sino que hay que transformar el valor pasándole
la función poly_reg (que guarda PolynomialFeatures) y el fit_transform. En definitiva,
tenemos que pasarle el valor transformado según PolynomialFeatures, ya que haremos una
predicción polinomial.
Si hago la predicción me daría un salario de 174K"""





















































