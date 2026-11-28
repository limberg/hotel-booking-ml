"""Entrenamiento de los modelos. Es el script principal del proyecto."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from hotel_booking_ml.config import MODELS_TESTS_DIR
from hotel_booking_ml.data_loader import (
    clean_data,
    load_raw_data,
    split_data,
)
from hotel_booking_ml.preprocessing import build_preprocessor


def get_models() -> dict:
    """Devuelve los modelos que se van a entrenar.

    Returns:
        dict: Nombre del modelo -> estimador de scikit-learn sin entrenar.
    """
    return {}


def save_model(modelo: Pipeline, nombre: str) -> Path:
    """Guarda un modelo entrenado en la carpeta models/tests.

    Args:
        modelo (Pipeline): Pipeline entrenado (preprocesado + modelo).
        nombre (str): Nombre del modelo, se usa como nombre del archivo.

    Returns:
        Path: Ruta del archivo .pkl guardado.
    """
    MODELS_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    ruta = MODELS_TESTS_DIR / f"{nombre}.pkl"
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en: {ruta}")

    return ruta


def train_model(
    nombre: str,
    estimador: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Entrena un modelo junto con el preprocesado y lo guarda.

    Args:
        nombre (str): Nombre del modelo.
        estimador (BaseEstimator): Modelo de scikit-learn sin entrenar.
        X_train (pd.DataFrame): Variables independientes de entrenamiento.
        y_train (pd.Series): Variable dependiente de entrenamiento.

    Returns:
        Pipeline: Pipeline entrenado (preprocesado + modelo).
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimador),
        ]
    )

    print(f"Entrenando {nombre}...")
    pipeline.fit(X_train, y_train)
    save_model(pipeline, nombre)

    return pipeline


def main() -> dict:
    """Lanza el flujo completo: carga, limpieza, split y entrenamiento.

    Returns:
        dict: Nombre del modelo -> Pipeline entrenado.
    """
    df = load_raw_data()
    df_limpio = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_limpio)

    modelos_entrenados = {}
    for nombre, estimador in get_models().items():
        modelos_entrenados[nombre] = train_model(
            nombre, estimador, X_train, y_train
        )

    return modelos_entrenados


if __name__ == "__main__":
    main()
