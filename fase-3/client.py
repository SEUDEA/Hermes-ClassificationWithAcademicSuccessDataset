import logging

from argparse import ArgumentParser

import pandas as pd

from requests import post


URL = "http://localhost:{port}/api/models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-5s - [%(filename)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def test_train(dataset: str, url: str) -> None:
    """
    Realiza pruebas al proceso de entrenamiento utilizando el dataset proporcionado.

    Este método envía una solicitud a la API para entrenar un modelo, primero sin 
    hiperparámetros y luego con hiperparámetros. El dataset se adjunta como un archivo 
    CSV en la solicitud. Si el entrenamiento es exitoso, se registra la respuesta; 
    de lo contrario, se lanza una excepción.

    Parameters:
    -----------
    dataset : str
        Ruta al archivo CSV que contiene los datos de entrenamiento.
    url : str
        URL base de la API a la que se enviarán las solicitudes de entrenamiento.

    Returns:
    --------
    None
    """
    log.info("Inician las pruebas al entrenamiento...")
    url = f"{url}/train"

    log.info("Inicia la prueba sin hiperparámetros")
    response = post(
        url,
        files={"dataset": ("train.csv", open(dataset, "rb"), "text/csv")},
    )

    if response.status_code != 201:
        log.info(f"Resultado: {response.json()}")
        raise Exception(f"Error en el API, Status Code: {response.status_code}")

    log.info("Entrenamiento exitoso")
    log.info(f"Resultado: {response.json()}")


    log.info("Inicia la prueba con hiperparámetros")
    response = post(
        url,
        params={"hyperparameter": True},
        files={"dataset": ("train.csv", open(dataset, "rb"), "text/csv")},
    )

    if response.status_code != 201:
        raise Exception(f"Error en el API, Status Code: {response.status_code}")

    log.info("Entrenamiento exitoso")
    log.info(f"Resultado: {response.json()}")

    log.info("Finalizan las pruebas de entrenamiento.")


def test_predict(dataset: str, url: str) -> None:
    """
    Realiza pruebas al proceso de predicción utilizando un subconjunto del dataset proporcionado.

    Este método selecciona aleatoriamente tres muestras del dataset y envía una solicitud 
    de predicción a la API para cada muestra. Si la predicción es exitosa, se registra 
    la respuesta; de lo contrario, se lanza una excepción.

    Parameters:
    -----------
    dataset : str
        Ruta al archivo CSV que contiene los datos de prueba para realizar las predicciones.
    url : str
        URL base de la API a la que se enviarán las solicitudes de predicción.

    Returns:
    --------
    None
    """
    log.info("Inician las pruebas a la predicción...")
    url = f"{url}/predict"

    test = pd.read_csv(dataset)
    sub_test = test.sample(3)

    for i, (_, row) in enumerate(sub_test.iterrows()):
        body = row.to_dict()

        response = post(url, json=body)
        if response.status_code != 200:
            raise Exception(f"Error en el API, Status Code: {response.status_code}")

        log.info(f"Predicción {i+1} exitosa")
        log.info(f"Resultado: {response.json()}")

    log.info("Finalizan las pruebas de predicción.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Pruebas automaticas al API")

    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Puerto en el que se encuentra desplegada el API.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nombre del archivo de datos, debe ser un csv.",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Tipo de prueba, las opciones son \"train\" o \"predict\"",
    )

    args = parser.parse_args()
    url = URL.format(port=args.port)

    if args.type == "train":
        test_train(args.dataset, url)
    elif args.type == "predict":
        test_predict(args.dataset, url)
    else:
        log.error("El tipo de prueba no es valido.")
