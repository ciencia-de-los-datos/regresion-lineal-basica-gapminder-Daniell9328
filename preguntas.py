"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv")

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df["life"]
    X = df["fertility"]

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.values.reshape(-1,1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.values.reshape(-1,1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv")

    # Imprima las dimensiones del DataFrame
    print(df.shape)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print(df['life'].corr(df['fertility']).round(4))

    # Imprima la media de la columna `life` con 4 decimales.
    print(df['life'].mean().round(4))

    # Imprima el tipo de dato de la columna `fertility`.
    print(type(df['fertility']))

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print(df['GDP'].corr(df['life']).round(4))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv")

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df["fertility"].values.reshape(-1,1)

    # Asigne a la variable los valores de la columna `life`
    y_life = df["life"].values.reshape(-1,1)

    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    reg = LinearRegression()

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
    ).reshape(-1,1)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_fertility, y_life)

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(reg.score(X_fertility, y_life).round(4))


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression
    # Importe train_test_split
    from sklearn.model_selection import train_test_split
    # Importe mean_squared_error
    from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("gm_2008_region.csv")

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df["fertility"]

    # Asigne a la variable los valores de la columna `life`
    y_life = df["life"]

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility, 
        y_life,
        test_size=0.8,
        random_state=53,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    linearRegression.fit(X_train, y_train)

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
    def pregunta_05():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe confusion_matrix
    from sklearn.metrics import confusion_matrix

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=gridSearchCV.predict(X_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=gridSearchCV.predict(X_test),
    )

    # Retorne la matriz de confusion de entrenamiento y prueba
    return cfm_train, cfm_test

def pregunta_06():
    """
    Pronóstico
    -------------------------------------------------------------------------------------
    """

    # Obtenga el pipeline de la pregunta 3.
    gridSearchCV = pregunta_04()

    # Cargue los datos generados en la pregunta 01.
    x_tagged, y_tagged, x_untagged, y_untagged = pregunta_01()

    # pronostique la polaridad del sentimiento para los datos
    # no etiquetados
    y_untagged_pred = gridSearchCV.predict(x_untagged)

    # Retorne el vector de predicciones
    return y_untagged_pred
