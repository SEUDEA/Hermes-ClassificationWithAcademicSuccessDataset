from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from schema import ModelParams, StudentData, Prediction


np.random.seed(42)


class Model:
    """
    Clase para gestionar un modelo de clasificación basado en KNeighborsClassifier, 
    incluyendo escalado de características, codificación de etiquetas, predicción 
    y entrenamiento del modelo.

    Parameters:
    -----------
    models_folder : str
        Ruta a la carpeta donde se almacenan los archivos del modelo, 
        escalador y codificador.

    Attributes:
    -----------
    sc : StandardScaler
        Escalador utilizado para normalizar las características de entrada.
    
    encoder : LabelEncoder
        Codificador utilizado para convertir las etiquetas de clase en 
        valores numéricos y viceversa.
    
    model : KNeighborsClassifier
        Modelo K-Nearest Neighbors cargado desde el archivo especificado.

    model_params : dict[str, Any]
        Diccionario que contiene los parámetros del modelo y el puntaje 
        obtenido durante el entrenamiento.
    
    models_folder : str
        Ruta a la carpeta donde se almacenan los archivos del modelo, 
        escalador y codificador.

    defaul_params : dict[str, Any]
        Diccionario con los parámetros predeterminados para el modelo 
        K-Nearest Neighbors.
    """

    def __init__(self, models_folder: str) -> None:
        """
        Inicializa la clase Model cargando el escalador, codificador y 
        modelo desde la carpeta especificada.

        Parameters:
        -----------
        models_folder : str
            Ruta a la carpeta que contiene los archivos 'scaler_knn_fase_3.joblib',
            'encoder_knn_fase_3.joblib' y 'knn_fase_3.joblib'.
        """
        self.sc: StandardScaler = load(f"{models_folder}/scaler_knn_fase_3.joblib")
        self.encoder: LabelEncoder = load(f"{models_folder}/encoder_knn_fase_3.joblib")
        self.model: KNeighborsClassifier = load(f"{models_folder}/knn_fase_3.joblib")

        params = self.model.get_params()
        self.model_params: dict[str, Any] = ModelParams(**params, score=None)

        self.models_folder = models_folder

        self.defaul_params = {
            "p": 1,
            "leaf_size": 20,
            "n_neighbors": 11,
            "algorithm": "auto",
            "weights": "uniform",
        }

    def predict(self, student: StudentData) -> Prediction:
        """
        Realiza una predicción utilizando el modelo cargado.

        Este método toma los datos de un estudiante, los transforma utilizando 
        el escalador y realiza una predicción con el modelo K-Nearest Neighbors. 
        La predicción es luego descodificada a su valor original utilizando el 
        codificador.

        Parameters:
        -----------
        student : StudentData
            Objeto que contiene los datos del estudiante para realizar la predicción.

        Returns:
        --------
        Prediction
            Objeto que contiene el id del estudiante y la decisión predicha.
        """
        x_predict = student.model_dump(exclude={"id"}, by_alias=True)
        x_predict_sc = self.sc.transform(pd.DataFrame([x_predict]))
        predictions = self.model.predict(x_predict_sc)
        decision = self.encoder.inverse_transform(predictions)
        return Prediction(id=student.id, decision=decision[0])

    def train(self, file: BytesIO, hyperparameter: bool = False) -> ModelParams:
        """
        Entrena el modelo K-Nearest Neighbors utilizando un conjunto de datos 
        proporcionado.

        Este método carga un conjunto de datos desde un archivo CSV, lo divide 
        en conjuntos de entrenamiento y validación, escala las características 
        y codifica las etiquetas. Si se especifica, realiza una búsqueda de 
        hiperparámetros; de lo contrario, utiliza los valores predeterminados. 
        Finalmente, el modelo entrenado, el escalador y el codificador se 
        guardan en la carpeta especificada.

        Parameters:
        -----------
        file : BytesIO
            Archivo CSV que contiene los datos para el entrenamiento del modelo.
        
        hyperparameter : bool, optional
            Si es `True`, realiza una búsqueda de hiperparámetros para 
            optimizar el modelo; si es `False`, utiliza los parámetros predeterminados.
            El valor predeterminado es `False`.

        Returns:
        --------
        ModelParams
            Objeto que contiene los parámetros del modelo y el puntaje obtenido 
            en el conjunto de validación.
        """
        dataset = pd.read_csv(file)
        x = dataset.drop("Target", axis=1)
        x = x.drop("id", axis=1)
        y = dataset["Target"]

        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y_encoded, test_size=0.2, random_state=42, shuffle=True,
        )

        self.sc = StandardScaler()
        x_train_sc = self.sc.fit_transform(x_train)
        x_val_sc = self.sc.transform(x_val)

        if hyperparameter:
            params = self.__hyperparameter_search(x_train_sc, y_train)
        else:
            params = self.defaul_params

        self.model = KNeighborsClassifier(**params)
        self.model.fit(x_train_sc, y_train)
        score = self.model.score(x_val_sc, y_val)
        self.model_params = ModelParams(**params, score=score)

        dump(self.sc, f"{self.models_folder}/scaler_knn_fase_3.joblib")
        dump(self.encoder, f"{self.models_folder}/encoder_knn_fase_3.joblib")
        dump(self.model, f"{self.models_folder}/knn_fase_3.joblib")
        return self.model_params

    def __hyperparameter_search(
        self,
        x_train_sc: np.ndarray,
        y_train: list[int],
    ) -> dict[str, Any]:
        """
        Realiza una búsqueda de hiperparámetros para el modelo K-Nearest Neighbors.

        Este método realiza una búsqueda en malla (GridSearchCV) sobre un conjunto 
        reducido al 50% de datos de entrenamiento para encontrar los mejores parámetros 
        del modelo. La búsqueda se realiza utilizando validación cruzada.

        Parameters:
        -----------
        x_train_sc : np.ndarray
            Conjunto de características de entrenamiento escaladas.
        
        y_train : list[int]
            Lista con las etiquetas de clase codificadas para el conjunto de 
            entrenamiento.

        Returns:
        --------
        dict[str, Any]
            Diccionario con los mejores hiperparámetros encontrados.
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
        return grid_search.best_params_
