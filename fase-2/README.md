# Fase 2

En esta segunda fase del proyecto, se requería encapsular dentro de un contenedor todo el código desarrollado en la Fase 1, separándolo en dos scripts principales. El primer script, `train.py`, se encarga de todo el proceso de entrenamiento, mientras que el segundo script, `predict.py`, se utiliza para realizar las predicciones. Además, se necesitaba crear un archivo `Dockerfile` para la construcción de la imagen del contenedor de **Docker**.

## Requerimientos

Para que el contenedor funcione correctamente, es necesario que los datos de la competición [*Classification with an Academic Success Dataset*](https://www.kaggle.com/competitions/playground-series-s4e6/overview) de Kaggle estén descomprimidos y ubicados en una carpeta llamada `storage`, la cual debe estar al mismo nivel que la carpeta `fase-2`. Asimismo, se requiere tener instalado **Docker** en su sistema operativo.

## Resultados

Al ejecutar los scripts siguiendo los comandos recomendados en la sección [Ejecutar](#ejecutar), se generarán 4 archivos. Tres de estos archivos se guardarán en la carpeta `storage/models/`, y el cuarto en la carpeta `storage`. Los archivos son:

- `knn_fase_2.joblib`: Modelo entrenado dentro del contenedor.
- `scaler_knn_fase_2.joblib`: Objeto *scaler* para los datos.
- `encoder_knn_fase_2.joblib`: Objeto *encoder* utilizado para codificar la variable objetivo.
- `predictions.csv`: Resultado de las predicciones, ubicado en la carpeta `storage`.

## Ejecutar

A continuación, se presentan una serie de comandos recomendados para la correcta ejecución del proyecto dentro del contenedor administrado por el *runtime* **Docker**. Si desea modificar los comandos para obtener un comportamiento diferente o renombrar archivos, es bajo su propia responsabilidad.

### Docker

Para crear la imagen del contenedor con **Docker**, ejecute el siguiente comando:

```zsh
docker image build -f Dockerfile -t hermes-fase-2:latest .
```

Este comando le indica a **Docker** que utilice el manifiesto de construcción para la imagen `Dockerfile`, que al crear la imagen, use el tag `hermes-fase-2:latest` para nombrarla y por ultimo que el contexto de creación es la misma carpeta desde donde se ejecuta el comando.

**Nota:** El `Dockerfile` utiliza una estrategia de compilación llamada *multi-stage* para optimizar el peso de la imagen y usar una imagen lo más ligera posible para la ejecución de la aplicación o scripts.

Para ejecutar la imagen previamente creada, utilice el siguiente comando:

```zsh
docker run -it --rm -v $PWD/../storage:/usr/src/app/storage hermes-fase-2:latest /bin/bash
```

Este comando le indica a **Docker** que ejecute un contenedor utilizando la imagen `hermes-fase-2:latest`, permitiendo la interacción directa con el contenedor mediante la bandera `-it`, eliminando el contenedor al finalizar su ejecución mediante la bandera `--rm`. Además, comparte un volumen, en este caso, la carpeta `storage`, para que esté disponible en el contenedor en `/usr/src/app/storage`, y el comando principal del contenedor será `/bin/bash`, que abre una terminal.

Una vez ejecutado este comando, estará dentro del contenedor y podrá ejecutar los scripts.

### Entrenamiento

Para ejecutar el script de entrenamiento, utilice el siguiente comando:

```zsh
python train.py --dataset ./storage/train.csv --model knn_fase_2.joblib --folder ./storage/models
```

Esto le indicará al script `train.py` que los datos de entrenamiento están en `./storage/train.csv`, que el modelo se llamará `knn_fase_2.joblib`, y que la carpeta para guardar el modelo y otros artefactos es `./storage/models`.

Este comando realizará un entrenamiento con parámetros predefinidos para el modelo. Si desea realizar una búsqueda de hiperparámetros, puede agregar la bandera `--hyperparameter_search` al comando anterior de la siguiente manera:

```zsh
python train.py --dataset ./storage/train.csv --model knn_fase_2.joblib --folder ./storage/models --hyperparameter_search
```

### Predicciones

Para ejecutar el script de predicciones, utilice el siguiente comando:

```zsh
python predict.py --model knn_fase_2.joblib --folder ./storage/models --input ./storage/test.csv --output ./storage/predictions.csv
```

Este comando le indicará al script `predict.py` que utilice el modelo `knn_fase_2.joblib` que se encuentra en la carpeta `./storage/models`, que los datos para la predicción están en `./storage/test.csv`, y que el resultado se guardará en `./storage/predictions.csv`.

---

Como el volumen se compartió con **Docker** y todos los archivos de salida se indicaron en la carpeta del volumen, aunque cierre el contenedor, los archivos no se perderán y quedarán en su sistema de archivos. Para cerrar o finalizar el contenedor, simplemente presione `Ctrl + D`, lo que detendrá el contenedor y lo eliminará, gracias a la bandera `--rm` utilizada en el comando inicial de **Docker**.