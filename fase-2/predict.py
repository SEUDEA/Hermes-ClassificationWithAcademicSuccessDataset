import logging
import sys

from argparse import ArgumentParser

import pandas as pd

from joblib import load

from utils import file_exists, file_ends_with_exntesion


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-5s - [%(filename)-10s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def predict_model(
    model_name: str,
    folder_name: str,
    input_name: str,
    output_name: str,
) -> None:
    """
    Realiza predicciones utilizando un modelo previamente entrenado y guarda los
    resultados en un archivo CSV.

    Esta función carga un modelo previamente entrenado junto con su escalador y
    codificador, aplica el modelo a un conjunto de datos de entrada, y guarda las
    predicciones en un archivo CSV especificado.

    Parameters:
    -----------
    model_name : str
        El nombre del archivo donde se encuentra guardado el modelo entrenado. 
        Este archivo debe estar en formato `.joblib`. El escalador y el codificador
        también se buscaran con los prefijos "scaler_" y "encoder_" respectivamente.
    
    folder_name : str
        El nombre de la carpeta donde se encuentran guardados el modelo, el escalador 
        y el codificador. La ruta completa a estos archivos se construye concatenando
        este nombre de carpeta con los nombres de archivo correspondientes.

    input_name : str
        El nombre y ruta del archivo CSV que contiene los datos de entrada para la
        predicción. Este archivo debe incluir una columna "id" que es el
        identificador de los estudiantes que se conservará en los resultados.

    output_name : str
        El nombre y ruta del archivo CSV donde se guardarán las predicciones. 
        El archivo de salida incluirá dos columnas: "id" y "prediction".

    Returns:
    --------
    None
        La función no retorna ningún valor. Guarda las predicciones en el archivo
        CSV especificado.

    Raises:
    -------
    FileNotFoundError
        Si cualquiera de los archivos especificados (modelo, escalador, codificador,
        o datos de entrada) no se encuentra en la ruta proporcionada.

    Example:
    --------
    >>> predict_model('knn_model.joblib', 'models', 'data_to_predict.csv', 'predictions.csv')
    >>> # Esto cargará el modelo 'knn_model.joblib' desde la carpeta 'models', 
    >>> # predecirá los resultados para 'data_to_predict.csv', y guardará las predicciones 
    >>> # en 'predictions.csv'.
    """
    log.info("Comienza la predicción...")
    x_predict = pd.read_csv(input_name)
    model = load(f"{folder_name}/{model_name}")
    sc = load(f"{folder_name}/scaler_{model_name}")
    encoder = load(f"{folder_name}/encoder_{model_name}")

    ids = x_predict["id"]
    x_predict = x_predict.drop("id", axis=1)
    x_predict_sc = sc.transform(x_predict)

    predictions = model.predict(x_predict_sc)
    predictions = encoder.inverse_transform(predictions)
    predictions = pd.DataFrame({"id": ids, "prediction": predictions})
    predictions.to_csv(output_name, index=False)
    log.info("Finaliza la predicción.")


def validate_arguments(
    model_name: str,
    folder_name: str,
    input_name: str,
    output_name: str,
) -> None:
    """
    Valida los argumentos de entrada para asegurar que cumplen con los requisitos
    esperados.

    Esta función verifica que los archivos y nombres de archivos proporcionados
    existen y tienen las extensiones correctas. Si alguno de los requisitos no se
    cumple, la función registra un error y termina la ejecución del script.

    Parameters:
    -----------
    model_name : str
        El nombre del modelo, que debe existir como un archivo con la extensión
        '.joblib'. Además, se verifica la existencia de archivos auxiliares con
        prefijos 'scaler_' y 'encoder_' seguidos del nombre del modelo.

    folder_name : str
        El nombre de la carpeta donde se espera encontrar el modelo, el escalador
        y el codificador. La función valida que estos archivos existan en la carpeta
        especificada.

    input_name : str
        El nombre del archivo de entrada, cuya existencia se valida.

    output_name : str
        El nombre del archivo de salida, que debe tener la extensión '.csv'.

    Returns:
    --------
    None
        No retorna ningún valor. Si alguna validación falla, se registra un error
        y se detiene la ejecución del script.

    Raises:
    -------
    SystemExit
        Si cualquiera de las validaciones falla, se detiene la ejecución del script
        con un código de salida `1`.

    Example:
    --------
    >>> validate_arguments('knn_model.joblib', 'models', 'test.csv', 'prediction.csv')
    >>> # Si 'test.csv' no existen, o 'knn_model.joblib' en la carpeta 'models' no existen,
    >>> # o 'prediction.csv' no tiene la extensión correcta, se mostrará un error y el 
    >>> # script se detendrá.
    """
    has_errors = False
    if not file_exists(f"{folder_name}/{model_name}") or \
        not file_ends_with_exntesion(model_name, ".joblib"):
        has_errors = True
    if not file_exists(f"{folder_name}/scaler_{model_name}"):
        has_errors = True
    if not file_exists(f"{folder_name}/encoder_{model_name}"):
        has_errors = True
    if not file_exists(input_name):
        has_errors = True
    if not file_ends_with_exntesion(output_name, ".csv"):
        has_errors = True

    if has_errors:
        log.error("Los argumentos del script son incorrectos.")
        sys.exit(1)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Cargar un modelo y hacer predicciones de los datos de un csv.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Nombre del archivo del modelo entrenado (formato .joblib)."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Nombre del volumen en donde se van a guardar el modelo entrenado."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Nombre del archivo csv con los datos para predecir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Nombre del archivo csv donde se guardarán las predicciones.",
    )

    args = parser.parse_args()
    validate_arguments(args.model, args.folder, args.input, args.output)
    predict_model(args.model, args.folder, args.input, args.output)
