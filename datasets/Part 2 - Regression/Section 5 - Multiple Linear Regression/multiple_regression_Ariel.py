#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:06:10 2020

@author: macbookariel
"""

# 16-05-20: 
# En este documento vamos a hacer la regresión lineal múltiple.
# Para ello primero copiamos la plantilla de pre procesado de datos y cambiamos
# lo que sea necesario

# Plantilla de Pre Procesado


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set, para este caso, se llama 50_Startups.
# Hay que recordar que hay que ubicarse en el directorio de trabajo.
# Escoger los datos a usar

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values #La variable a predecir suele ubicarse como la última,
                                # así que esta línea no cambia. Los valores de X
                                # están en el resto de columnas.
                                
y = dataset.iloc[:, 4].values   # Ahora bien, la variable a predecir ahora 
                                # es la columna 4 (recordar empezar por cero)


# El siguiente paso es codificar la columna categórica
# que contiene los estados

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)

# Luego de transformar la variable estado a variable dummy, es posible
# que las columnas cambien de lugar así que hay que tenerlo en cuenta

# Evitar la trampa de las variables ficticias. Es decir que debemos eliminar
# una variable ficticia (en general la primera, la cero),
# en este caso, guardamos en X todas las filas,
# pero solamente las columnas de la 1 en adelante.

X = X[:, 1:]
# Ahora vamos a ejectura y ver si se ha eliminado la columna


# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)



# Escalado de variables: NO ES NECESARIO EN ESTE CASO
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# El siguiente paso sería ajustar el modelo de regresión lineal múltiple
# con el conjunto de entrenamiento. Observar que usamos el método fit

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

# Aquí el modelo ya estaría ajustado con los datos de entrenamiento
# El ordenador ya ha aprendido lo que necesita, así que ahora vamos a aplicarlo
# al conjunto de testing

"""Predicción de los resultado en el conjunto de testing
# Vamos a crear un objeto de vector de predicción y en este caso
# vamos a usar el objeto regression que creamos previamente pero con el método
predict"""

y_pred = regression.predict(X_test)

# RLM con eliminación hacia atrás. Primer paso: preparar el sistema. Construir el modelo
# Vamos a importar las librerías que necesitamos

import statsmodels.api as sm

# Según nuestra ecuación de la regresión (y = termino independiente + X1b1 + X2b2...),
# cada variable independiente tiene un 
# coeficiente y tenemos todas las variables representadas en nuestra X.
# Sin embargo, no sabemos si lo que se puede eliminar es el término independiente
# de la ecuación y no tenemos forma de conocerlo, así que vamos a agregar una
# columna más solo compuesta de unos y que representará la ordenada al origen.
# Luego buscaremos el pvalor de todas las columnas y veremos si el término
# independiente (columna de unos) se puede eliminar o no.

# Usaremos numpy y append (agregar) al array que tenemos en X un método
# que se llama np.ones y que agregará unos. 
#En este caso como tenemos 50 filas y 1 columna
# entonces agregamos 50,1. 
# Por defecto agregará unos en formato float así que los cambiamos a enteros
# usando astype(int) y luego le decimos que los queremos todo en columna
# usando axis =1. Quedaría:

# X = np.append(arr = X, values = np.ones((50,1)). astype(int), axis = 1)

# El problema de la línea anterior es que se agregarían como columna final de X
# Por ello, si queremos que se agreguen al inicio cambiamos los datos


X = np.append (arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# Vamos a crear una variable que vaya captando lo versión óptima,
# es decir, las variables estadísticamente significativas. Partiremos de todas
# las variables y en cada paso irá desapareciendo una columna.
# En esta nueva variable X_opt se formará con todas las filas de X y 
# forzaremos a ir cogiendo todas las columnas, desde la 0 a la 5

X_opt = X[:, [0,1,2,3,4,5]]

#Luego indicaremos el nivel de significatividad para que la variable se quede.

SL = 0.05

# En este caso vamos a tener que volver a generar un modelo porque la librería
# statsmodels lo requiere con todos los nuevos cambios. El objeto se llama OLS
# ordinary least squares. Vamos a implementar dicha librería en un objeto
# llamado regression_OLS que necesita dos variables y se ajusta directamente
# con .fit

regression_OLS = sm.OLS (endog = y, exog = X_opt).fit()

# Ya está creado el modelo y ahora hay que ver cómo se consulta el pvalue,
# con summary

regression_OLS.summary()

# Tras ver el resultado, vamos a eliminar la variable con el pvalor más elevado
# En este caso, el pvalor más elevado es la variable x2, que correspondía a la
# variable dummy que representaba al estado de NY. La eliminaremos y ajustaremos
# de nuevo el modelo.

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS (endog = y, exog = X_opt).fit()
regression_OLS.summary()

# En este caso, la variable con el pvalor más alto es la x1, dentro de la 
# variable X_opt y correspondería eliminar la columna 1


X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS (endog = y, exog = X_opt).fit()
regression_OLS.summary()


X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS (endog = y, exog = X_opt).fit()
regression_OLS.summary()


X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS (endog = y, exog = X_opt).fit()
regression_OLS.summary()

# Después de todas eliminaciones teniendo en cuenta el pvalor, al final nos
# quedamos con una sola variable

# Eliminación hacia atrás automatizada usando solo pvalor

import statsmodels.formula.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Eliminación hacia atrás utilizando  p-valores y 
#el valor de  R Cuadrado Ajustado:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



















































































