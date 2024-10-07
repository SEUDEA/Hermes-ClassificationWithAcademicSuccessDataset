import logging
import sys

from argparse import ArgumentParser
from typing import Any

import pandas as pd
import numpy as np

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from utils import file_exists, file_ends_with_exntesion


np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-5s - [%(filename)-10s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def train_model(
    dataset: str,
    model_name: str,
    folder_name: str,
    hyperparameter: bool = False,
) -> None:
    """
    Entrena un modelo K-Nearest Neighbors (KNeighborsClassifier) utilizando
    un conjunto de datos especificado.

    Este método carga un conjunto de datos desde un archivo CSV, preprocesa
    los datos, realiza una búsqueda de hiperparámetros opcional, entrena el
    modelo, y guarda el modelo entrenado, el escalador y el codificador 
    en archivos específicos dentro de la carpeta proporcionada.

    Parameters:
    -----------
    dataset : str
        La ruta al archivo CSV que contiene el conjunto de datos de entrenamiento.
        El archivo debe contener una columna llamada "Target" que representa la
        variable objetivo, y una columna "id" que será el identificador de los
        estudiantes que será eliminada antes del entrenamiento.

    model_name : str
        El nombre del archivo donde se guardará el modelo entrenado.
        El escalador y el codificador también se guardarán con prefijos
        "scaler_" y "encoder_" respectivamente.

    folder_name : str
        El nombre de la carpeta donde se guardarán el modelo, el escalador 
        y el codificador. La ruta completa a estos archivos se construye
        concatenando este nombre de carpeta con los nombres de archivo
        correspondientes.

    hyperparameter : bool
        Si es `True`, se realiza una búsqueda de hiperparámetros para optimizar
        el modelo. Si es `False`, se usan parámetros predefinidos.

    Returns:
    --------
    None
        La función no retorna ningún valor. Los resultados incluyen un
        modelo entrenado, un escalador, y un codificador guardados en archivos.

    Raises:
    -------
    FileNotFoundError
        Si el archivo CSV especificado no se encuentra en la ruta proporcionada.

    Example:
    --------
    >>> train_model('train.csv', 'knn_model.joblib', 'models', hyperparameter=True)
    >>> # Esto entrenará el modelo usando 'train.csv', realizará una búsqueda de
    >>> # hiperparámetros, y guardará el modelo entrenado en la carpeta 'models'
    >>> # como 'knn_model.joblib', junto con el escalador y el codificador.
    """
    log.info("Comienza el entrenamiento...")

    train = pd.read_csv(dataset)

    x = train.drop("Target", axis=1)
    x = x.drop("id", axis=1)
    y = train["Target"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, shuffle=True,
    )

    sc = StandardScaler()
    x_train_sc = sc.fit_transform(x_train)
    x_val_sc = sc.transform(x_val)

    if hyperparameter:
        log.info("Realizando búsqueda de hiperparámetros...")
        params = hyperparameter_search(x_train_sc, y_train, x_val_sc, y_val)
    else:
        log.info("Usando parámetros predefinidos...")
        params = {
            "algorithm": "auto",
            "leaf_size": 20,
            "n_neighbors": 11,
            "p": 1,
            "weights": "uniform",
        }

    log.info(f"Entrenando el modelo KNeighborsClassifier con los parametros: \n{params}")
    model = KNeighborsClassifier(**params)
    model.fit(x_train_sc, y_train)
    score = model.score(x_val_sc, y_val)
    log.info(f"Score del modelo entrenado: {score}")

    log.info("Guardando el modelo...")
    dump(model, f"{folder_name}/{model_name}")
    dump(sc, f"{folder_name}/scaler_{model_name}")
    dump(encoder, f"{folder_name}/encoder_{model_name}")

    log.info("Finaliza el entrenamiento.")


def hyperparameter_search(
    x_train_sc: np.ndarray,
    y_train: list[int],
    x_val_sc: np.ndarray,
    y_val: list[int],
) -> dict[str, Any]:
    """
    Realiza una búsqueda de hiperparámetros utilizando `GridSearchCV` para optimizar un
    modelo K-Nearest Neighbors (KNeighborsClassifier).

    Esta función toma datos preprocesados de entrenamiento y validación, define un
    conjunto de hiperparámetros para explorar, y utiliza validación cruzada para encontrar
    la mejor combinación de hiperparámetros que maximiza la precisión del modelo.
    Devuelve los mejores hiperparámetros encontrados.

    Parameters:
    -----------
    x_train_sc : np.ndarray
        Array de características escaladas para el conjunto de entrenamiento.
    y_train : list[int]
        Lista de etiquetas correspondientes al conjunto de entrenamiento.
    x_val_sc : np.ndarray
        Array de características escaladas para el conjunto de validación.
    y_val : list[int]
        Lista de etiquetas correspondientes al conjunto de validación.

    Returns:
    --------
    dict[str, Any]
        Un diccionario que contiene los mejores hiperparámetros encontrados durante
        la búsqueda, en la forma `{"param_name": best_value}`.

    Example:
    --------
    >>> best_params = hyperparameter_search(x_train_sc, y_train, x_val_sc, y_val)
    >>> print(best_params)
    {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2}

    Notes:
    ------
    - La búsqueda se realiza usando una versión reducida del conjunto de entrenamiento
        (50%) para acelerar el proceso.
    - La validación cruzada se realiza con 5 particiones (`cv=5`).
    - El `scoring` usado para evaluar el modelo es la `accuracy`.
    """
    x_train_sc_small, _, y_train_small, _ = train_test_split(
        x_train_sc, y_train, train_size=0.5, random_state=42, stratify=y_train,
    )

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2]
    }

    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(x_train_sc_small, y_train_small)
    score = grid_search.best_estimator_.score(x_val_sc, y_val)
    log.info(f"Score del mejor modelo encontrado en la búsqueda de hiperparámetros: {score}")
    return grid_search.best_params_


def validate_arguments(dataset: str, model_name: str) -> None:
    """
    Valida los argumentos proporcionados para asegurar que son correctos y cumplen
    con los requisitos del script.

    Esta función verifica que el archivo de datos especificado exista y que el nombre
    del archivo del modelo tenga la extensión correcta. Si alguna de las validaciones
    falla, se registra un error y se finaliza la ejecución del script.

    Parameters:
    -----------
    dataset : str
        La ruta al archivo CSV que contiene el conjunto de datos. La función verifica
        que el archivo exista.
    
    model_name : str
        El nombre del archivo donde se guardará el modelo entrenado. Debe tener la
        extensión `.joblib`.

    Returns:
    --------
    None
        La función no retorna ningún valor. En caso de error en las validaciones,
        termina la ejecución del script.

    Raises:
    -------
    SystemExit
        Si cualquiera de las validaciones falla, se detiene la ejecución del script
        con un código de salida `1`.

    Example:
    --------
    >>> validate_arguments('train.csv', 'knn_model.joblib')
    >>> # Si 'train.csv' no existe o 'knn_model.joblib' no tiene la extensión correcta,
    >>> # se mostrará un error y el script se detendrá.
    """
    has_errors = False
    if not file_exists(dataset):
        has_errors = True
    if not file_ends_with_exntesion(model_name, ".joblib"):
        has_errors = True
    
    if has_errors:
        log.error("Los argumentos del script son incorrectos.")
        sys.exit(1)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Entrenamiento de un modelo con opciones de hiperparámetros.",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nombre del archivo de datos de entrenamiento, debe ser un csv.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Nombre del volumen en donde se van a guardar el modelo entrenado.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Nombre del archivo para guardar el modelo entrenado."
            "debe tener la extensión joblib."
        ),
    )
    parser.add_argument(
        "--hyperparameter_search",
        action="store_true",
        help=(
            "Si se activa, se realiza la búsqueda de hiperparámetros."
            "Si no, se utilizan los valores predefinidos."
        ),
    )

    args = parser.parse_args()
    validate_arguments(args.dataset, args.model)
    train_model(args.dataset, args.model, args.folder, args.hyperparameter_search)
