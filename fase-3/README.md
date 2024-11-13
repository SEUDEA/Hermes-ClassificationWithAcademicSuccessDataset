# Fase 3

En esta tercera fase del proyecto, se requería exponer el proceso de entrenamiento y predicción del modelo ya configurado a través de una API RESTful. Para ello, se utilizó **FastAPI** junto con **Pydantic** para aprovechar la documentación automática proporcionada por FastAPI.

La API se encapsuló dentro de un contenedor de **Docker** y se utilizó un archivo `docker-compose.yml` para facilitar el encendido y apagado de la API. El `Dockerfile` utilizado es el mismo que en la Fase 2, con la adición del export al puerto 80 para permitir las conexiones a la API.

## Requerimientos

Para que el contenedor funcione correctamente, es necesario que los datos de la competición [*Classification with an Academic Success Dataset*](https://www.kaggle.com/competitions/playground-series-s4e6/overview) de Kaggle estén descomprimidos y ubicados en una carpeta llamada `storage`, la cual debe estar al mismo nivel que la carpeta `fase-3`. Asimismo, se requiere tener instalado las ultimas versiones de **Docker**, asegurando que contenda los comando de **Docker Compose**, en su sistema operativo.

## Resultados

Al consumir el endpoint de entrenamiento `/api/models/train`, se generan los mismos 3 archivos de la Fase 2, pero adaptados para esta fase:

- `knn_fase_3.joblib`: Modelo entrenado dentro del contenedor.
- `scaler_knn_fase_3.joblib`: Objeto *scaler* para los datos.
- `encoder_knn_fase_3.joblib`: Objeto *encoder* utilizado para codificar la variable objetivo.

Al consumir el endpoint de predicción `/api/models/predict`, la API retorna el resultado de la predicción para la información proporcionada.

## Ejecutar

A continuación, se presentan los pasos recomendados para la correcta ejecución del proyecto dentro del contenedor administrado por **Docker**, utilizando **Docker Compose**.

### Docker Compose

Para crear y levantar los contenedores, ejecute el siguiente comando:

```zsh
docker compose -f docker-compose.yml up --build
```

Este comando construirá la imagen y levantará el contenedor que expone la API en el puerto 8080. Una vez ejecutado, puede acceder a la documentación automática de **FastAPI** en su navegador web mediante la URL: http://127.0.0.1:8080/docs#/

**Nota: En caso de modificar el archivo `docker-compose.yml` para cambiar el puerto, se debe cambiar el puerto de la URL anterior.**

En esta documentación, podrá interactuar con todos los endpoints disponibles de la API.

### Endpoints de la API

1. **Entrenamiento:**
   - **Endpoint:** `/api/models/train`
   - **Método:** POST
   - **Parámetros:**
     - `hyperparameter` (opcional, boolean): Si se desea realizar una búsqueda de hiperparámetros, se pasa como un parámetro de consulta (query param).
   - **Body:** Se debe proporcionar un dataset para entrenamiento (se recomienda utilizar el dataset de Kaggle mencionado anteriormente).
   - **Resultado:** Retorna los parámetros y la puntuación del modelo entrenado.

2. **Predicción:**
   - **Endpoint:** `/api/models/predict`
   - **Método:** POST
   - **Body:** Recibe la información de un estudiante, equivalente a una fila del dataset de test de la competición de Kaggle.
   - **Resultado:** Retorna la predicción y el ID del estudiante.

3. **Configuración del modelo actual:**
   - **Endpoint:** `/api/models/current`
   - **Método:** GET
   - **Resultado:** Retorna los parámetros del modelo actual y la puntuación del entrenamiento (si está disponible).

### Ejecutar Cliente

Para probar los principales endpoints de la API, se ha incluido un script llamado `client.py` que contiene dos funciones principales:

1. Función test_train: Permite probar el proceso de entrenamiento del modelo.
2. Función test_predict: Permite probar el proceso de predicción utilizando datos de prueba.

Para ejecutar el script y probar el entrenamiento del modelo, utiliza el siguiente comando:

```zsh
python client.py --port 8080 --type train --dataset ../storage/train.csv
```

Y para probar las predicciones, utiliza:

```zsh
python client.py --port 8080 --type predict --dataset ../storage/test.csv
```

Es importante destacar que este script no se ejecuta dentro del contenedor de Docker, sino que debe ser ejecutado en el entorno principal del proyecto. Se recomienda tener instalado Python 3.11 (aunque otras versiones también pueden funcionar, bajo la responsabilidad del usuario) y las bibliotecas necesarias, que se pueden instalar con el siguiente comando:

```zsh
pip install pandas requests
```

En los comandos anteriores:

* La bandera `--port` se utiliza para indicar el puerto en el que se está ejecutando la API. Por defecto, se ha configurado en el puerto 8080, pero si se modifica el archivo `docker-compose.yml` para cambiar el puerto, se debe ajustar el comando correspondiente.
* La bandera `--type` indica el endpoint que se desea probar (train para entrenamiento y predict para predicciones).
* La bandera `--dataset` especifica el archivo CSV que contiene los datos a utilizar. Para entrenamiento, se recomienda usar el archivo `train.csv`, y para predicciones, el archivo `test.csv`. Durante las pruebas de predicción, se seleccionan 3 valores aleatorios del dataset para realizar las pruebas.

Los archivos mencionados (`train.csv` y `test.csv`) deben estar ubicados en la carpeta `storage`, la cual debe encontrarse al mismo nivel que la carpeta `fase-3`.


### Apagado del Proyecto

Una vez finalizada la ejecución del proyecto, se recomienda detener y eliminar el contenedor y la red interna creados por **Docker Compose**, utilizando el siguiente comando:

```zsh
docker compose -f docker-compose.yml down
```

Este comando garantiza que no queden contenedores o redes en ejecución innecesarias después de usar la API.
