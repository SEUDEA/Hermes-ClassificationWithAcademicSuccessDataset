import logging
import os


log = logging.getLogger(__name__)


def file_exists(filename: str) -> bool:
    """
    Verifica si un archivo existe en la ruta especificada.

    Esta función comprueba si el archivo especificado por
    `filename` existe en el sistema de archivos. Si el archivo
    no existe, registra un error y retorna `False`. Si el archivo
    existe, retorna `True`.

    Parameters:
    -----------
    filename : str
        La ruta completa o nombre del archivo que se desea verificar.

    Returns:
    --------
    bool
        Retorna `True` si el archivo existe, de lo contrario retorna
        `False` y registra un mensaje de error.
    """
    if not os.path.exists(filename):
        log.error(f"El archivo {filename} no existe.")
        return False
    return True


def file_ends_with_exntesion(filename: str, suffix: str) -> bool:
    """
    Verifica si un archivo tiene una extensión específica.

    Esta función comprueba si el nombre de archivo termina con la extensión
    especificada por `suffix`. Si el archivo no tiene la extensión correcta,
    registra un error y retorna `False`. Si la extensión  es correcta,
    retorna `True`.

    Parameters:
    -----------
    filename : str
        El nombre del archivo que se desea verificar.

    suffix : str
        La extensión que se espera que tenga el archivo
        (por ejemplo, '.csv', '.joblib').

    Returns:
    --------
    bool
        Retorna `True` si el archivo tiene la extensión especificada,
        de lo contrario retorna `False` y registra un mensaje de error.
    """
    if not filename.endswith(suffix):
        log.error(f"El archivo {filename} no tiene la extensión {suffix}.")
        return False
    return True
