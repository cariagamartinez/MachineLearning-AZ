#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:15:29 2020

@author: macbookariel
"""
# **Vamos a importar todas las librerías que vamos a usar y le damos un alias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Guardamos el data set en una variable que se llama "dataset".
# Usamos panda pd.read para que "lea" los datos guardados en el archvivo csv.
# Siempre hay que ver arriba que estemos en el mismo fichero de trabajo donde el csv.

dataset = pd.read_csv("Data.csv")

# Vamos a crear la variables independientes (X) y la dependiente.

# Para esas variables vamos a seleccionar los datos dentro de dataset e indicar cuál es cuál.

# Para ello usamos nombre_de_variable.iloc. Quedaría:

# dataset.iloc

# Y entre corchetes las FILAS y las COLUMNAS QUE DESEAMOS.

# Si queremos todas las filas, entonces usamos : sin más.

# Luego una coma.

# Luego todas las columnas que querramos. En este caso queremos todas, excepto la última.

# Luego .values para coger los valores de la matriz. Por eso la llamamos X mayúscula.

X = dataset.iloc[:, :-1].values

# En el scrip no vamos a ver lo que se carga aquí, pero podemos ejecutar X en la consola y ver qué contiene.

# Ahora vamos a crear la variable dependiente. Como es una variable sin mas llamamos en minúsculas.

# Y repetimos el proceso anterior, pero indicando que la columna que buscamos es solo la cuarta.

# Recordemos que en Python empezamos en cero, así que la cuarta columna es la número 3.

y = dataset.iloc[:, 3].values

# Tratamiento de los NAs. Se hará con sklearn. Pero no vamos a importar toda la librería, sino solo la función que nos interesa.

# En este caso, vamos a usar la librería impute y dentro de ella, la función SimpleImputer.

# Lo que queremos hacer ahora es reemplazar todos los valores NaN, a una media, tomando como base la columna.


from sklearn.impute import SimpleImputer

# Vamos crear una variable imputer, luego llamaremos a la clase simple.imputer, la cual tiene parámetros, como missing_values que,

# en este caso son NaN y colocaremos la media de toda la columna. Seleccionar columnas se hace con axis = 0. Seleccionar filas sería axis = 1

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

# Vamos a hacer un fit de nuestra variable "imputer"y para ello usamos el .fit y el objeto a arreglar.

# En este caso se trata de la X, y de sus elementos, queremos arreglar todas las filas (por eso usamos :) y solamente las columnas 1 y 2.

# En Python empezamos a contar desde cero y el último número no se tiene en cuenta, así que si quiero evitar la columna "País" (Columna 0)

# pero quiero usar el resto de columnas, entonces pondré los número 1:3 (es decir columna 1 y 2, el ultimo número se obvia).

imputer = imputer.fit(X[:, 1:3])

# Ahora vamos a arreglar la X, usando la función transform, que finalmente cambiará los datos según el SimpleImputer 

# (recordemos que a SimpleImputer le pedíamos las medias)

X[:, 1:3] = imputer.transform(X[:, 1:3])


# ** Vamos a codificar (traducir a números) los datos categóricos.

# Para ello necesitamos una librería que a cada valor categórico le asigne un valor numérico. Vamos a importar la librería necesaria.

from sklearn import preprocessing

# Vamos a crear un label encoder (le: codificador de etiquetas) para X llamando a la función preprocessing.LabelEncoder.

# Esta función no requiere ningún parámetro, por eso tenemos paréntesis vacíos.

le_X = preprocessing.LabelEncoder()

# Ahora vamos a transformar las columnas a un dato numérico, indicando qué filas vamos a tomar (todas = :) y qué columnas vamos a tomar

# la primera columna 

X[:, 0] = le_X.fit_transform(X[:, 0])

# Ahora, si ejecutamos la X veremos que a cada país se le ha asignado un número. Sin embargo, esto puede dar lugar a un confusión 

# y es creer que esos números son realmente números y no índices (1 representa Francia y 2 España, pero no significa que 1 sea mayor que 2)

# Para evitarlo, vamos a crear variables dummy, que obtendrán un valor 1 cuando coincidan con su string:

# Así: Francia   España   Alemania
#       1         0        0        = Significa Francia
#       0         1        0        = Significa España
# etc.

# Para ello utilizaremos dos librerías nuevas:

# Codificar datos categóricos

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Aquí creamos una variable labelencoder_X que llama a la función LabelEncoder.
# Luego indicamos que en todas las filas y en primera columna de X, el labelenconder se modifique así mismo en dichos puntos
# y vuelva a ser guardado en los mismos puntos.

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Ahora vamos a definir una variable (ct) que llamará a la función ColumnTransformer
# La función ColumnTransformer tiene al menos un parámetro (que se llama transformers) que requiere tres argumentos: 
# un nombre, el transformador a aplicar (en este caso OneHotEncoder y su parámetro)
# La columna a transformar

ct = ColumnTransformer(
    [('aqui_va_cualquier_nombre', OneHotEncoder(categories='auto'), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

# Y luego transformamos la variable ct (a la que se aplicó el fit_transform en un array de numpy de tipo float)
X = np.array(ct.fit_transform(X), dtype=np.float)

#Ahora vamos a hacer lo mismo con la variable y. En este caso, solo tenemos "yes" y "no", así que le daremos un 0 a "no" y un "1" a sí.
# Por ello, hacemos un labelencoder_y que llame a la función LabelEncoder() 
# y luego guardamos en la variable y, el mismo labelencoder que se ha modificado a sí mismo
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Ahora vamos a dividir el dataset en conjunto de entrenamiento del modelo (70-80% del total de datos) y conjunto de testing (20%)
# Para ello vamos a usar una librería que divide los conjuntos y divide las dos variables x e y en 4:
# X de entrenamiento y X de test y luego Y de entrenamiento e y de test.
# Una vez indicados los nombres de las variables, llamamos a la función train_test_split y le indicamos las variables originales: X e y.
# Luego indicamos la proporción de datos que se destinará a test. En este caso el 20% (test_size = 0.2)
# Finalmente le diremos que que inicie de modo aleatorio el reparto de datos. Pero le daremos una semilla (random_state) 
# Si usamos la misma semilla (puede ser cualquier entero) siempre obtendremos la misma división de los datos.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Ahora tendríamos que "escalar" los valores, de tal forma que las magnitudes de las mediciones sean comparables.
# En nuestro caso, la variable "edad" va de 20 a 50 años, pero la variable sueldo va de 48,000 a 90,000.
# con lo cual, si usamos directamente los datos, la contribución del primero será ínfima frente al segundo.
# Podríamos estandarizar (x-media / DS) y generaríamos una campana de Gauss, con muchos valores cercanos a cero.
# Y si un valor es muy pero muy grande llegaría a 2-3.
# o podríamos normalizar (x - valor mínimo de x / maximo - mínimo -rango-) lo que nos daría valores máximos de 1 y 0 y el resto
# de valores se escalaria linealmente.

# Escalado de variables. Vamos a importar la función StandardScales que estandarizará la variable

from sklearn.preprocessing import StandardScaler

# Aquí vamos a crear una variable sc_X que llamará a StandardScales.
# Luego vamos a decirle que pase a la variable  X_train un sc_X un fit_transform a sí mismo.
# Luego usamos esa misma forma de escalado sobre la variable X_test, invocando a transform.

# Me he dado cuenta de que si leo al revés, me entero mejor. Para el siguiente código:

# StandardScaler va a ser asignada a una función que se llama sc_X
# Luego llamaremos a sc_X (que es StandardScaler realmente) y le diremos que haga un fit_transform a la variable X_train
# y vuelva a guardar los cambios en la variables X_train (ahora con nuevos datos)

# En este caso, el método fit_transform hace la estandarización y reemplaza los datos.
# luego llamaremos a la variable sc_X (que realmente invoca a StandarScaler) 
# y usaremos el método transform que hará una estandarización igual sobre la variable X_test y la volverá a guardar en X_test

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)













