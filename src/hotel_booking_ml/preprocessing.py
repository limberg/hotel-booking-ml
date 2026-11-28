"""Preprocesado de las variables: imputación, escalado y one-hot."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from hotel_booking_ml.config import (
    ADR_MAX,
    ADR_MIN,
    BINARY_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    ONEHOT_MIN_FREQUENCY,
)


def build_preprocessor() -> ColumnTransformer:
    """Construye el transformador de columnas, sin ajustar todavía.

    Returns:
        ColumnTransformer: Transformador con cuatro bloques: numéricas,
        adr, binarias y categóricas.
    """
    # adr tiene su propio bloque, así que lo quitamos de las numéricas
    numeric_cols = [col for col in NUMERIC_COLS if col != "adr"]

    # Numéricas: imputamos con la mediana y estandarizamos
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # adr: igual que las numéricas, pero antes de escalar recortamos los
    # valores fuera de rango (clipping) a los límites de la limpieza.
    # En train no cambia nada (clean_data ya los quitó): es un "por si acaso"
    # para reservas nuevas con un adr negativo o exagerado
    adr_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clipper",
                FunctionTransformer(
                    np.clip,
                    kw_args={"a_min": ADR_MIN, "a_max": ADR_MAX},
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # Categóricas: imputamos con el valor más frecuente y aplicamos one-hot
    # Las categorías con menos reservas que ONEHOT_MIN_FREQUENCY se agrupan
    # en una sola
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    min_frequency=ONEHOT_MIN_FREQUENCY,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("adr", adr_pipeline, ["adr"]),
            ("binary", "passthrough", BINARY_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
