# Fase 1

En esta primera fase del proyecto, se requería seleccionar una competición de Kaggle y desarrollar un notebook capaz de entrenar un modelo y utilizarlo para realizar predicciones. Para mi caso, opté por la competición [*Classification with an Academic Success Dataset*](https://www.kaggle.com/competitions/playground-series-s4e6/overview). El notebook que desarrollé no es completamente de mi autoría, ya que me basé en varios códigos disponibles en la sección *Code* de la competición. Sin embargo, no copié y pegué código de otros; aunque es posible que algunas ideas o enfoques se asemejen a los que observé en esas respuestas.

Dado que esta primera fase simula una prueba de concepto, el notebook ha sido nombrado `Prueba de Concepto.ipynb`.

## Requerimientos

Para que el notebook funcione correctamente, es necesario que los datos de la competición estén descomprimidos y ubicados en una carpeta llamada `storage`, que debe estar al mismo nivel que la carpeta `fase-1`. Además, se requiere un entorno de desarrollo que utilice `Python 3.11`. Aunque podría funcionar con otras versiones de **Python**, se recomienda usar esta versión, ya que fue la utilizada durante el desarrollo. **Si decides usar otra versión, será bajo tu propia responsabilidad.**

## Resultados

El notebook generará dos archivos dentro de la carpeta `storage/models/`: el primero, `knn_fase_1.joblib`, es el modelo entrenado; y el segundo, `scaler_fase_1.joblib`, es el objeto *scaler* para los datos.

## Estructura del Notebook

El notebook se divide en dos secciones principales, precedidas por un fragmento inicial de configuración.

1. **Fragmento de Configuración:** Este fragmento se encarga de instalar las bibliotecas necesarias para asegurar el correcto funcionamiento del resto del notebook.

2. **Sección de Entrenamiento:** En esta sección se entrena el modelo, generando los archivos mencionados en los [resultados](#resultados). Esta sección debe ejecutarse al menos una vez para poder utilizar la siguiente sección.

3. **Sección de Predicción:** Esta sección está diseñada para probar las predicciones del modelo, asumiendo que el modelo ya ha sido guardado en los archivos correspondientes. Aquí, el notebook carga esos archivos y los utiliza directamente para realizar predicciones.
