import os

from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schema import ModelParams, StudentData, Prediction
from model import Model


model = Model(os.getenv("MODEL_FOLDER", "./storage/models"))

app = FastAPI(
    title="Proyecto Hermes para Modelados y Simulación I",
    description=(
        "Esta API realiza la predicción de decisiones estudiantiles "
        "y permite reentrenar el modelo utilizado para dichas "
        "predicciones. Forma parte del Proyecto Hermes, el cual es "
        "el proyecto final de la materia Modelos y Simulación I."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post(
    "/api/models/predict",
    tags=["Models"],
    response_class=JSONResponse,
    response_model=Prediction,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful prediction"},
    },
)
async def predict(student: StudentData):
    try:
        prediction = model.predict(student)
        return prediction
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post(
    "/api/models/train",
    tags=["Models"],
    response_class=JSONResponse,
    response_model=ModelParams,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Successful training"},
    },
)
async def train(dataset: Annotated[bytes, File()], hyperparameter: bool = False):
    try:
        params = model.train(BytesIO(dataset), hyperparameter)
        return params
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get(
    "/api/models/current",
    tags=["Models"],
    response_class=JSONResponse,
    response_model=ModelParams,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Model params found"},
    },
)
async def get_params():
    return model.model_params
