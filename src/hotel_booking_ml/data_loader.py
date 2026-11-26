"""Carga, limpieza, partición y preprocesado de los datos."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_booking_ml.config import (
    ADR_MAX,
    ADR_MIN,
    AGENT_MISSING_LABEL,
    BINARY_COLS,
    CATEGORICAL_COLS,
    CHILDREN_FILL_VALUE,
    COMPANY_COL,
    COUNTRY_MISSING_LABEL,
    FEATURE_COLS,
    HAS_COMPANY_COL,
    LEAKAGE_COLS,
    MIN_GUESTS,
    MIN_NIGHTS,
    NUMERIC_COLS,
    ONEHOT_MIN_FREQUENCY,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
)


def load_raw_data() -> pd.DataFrame:
    """Carga el CSV original sin modificar nada.

    Returns:
        pd.DataFrame: DataFrame con las reservas tal y como vienen en el CSV.
    """
    return pd.read_csv(RAW_DATA_PATH)


def agent_a_texto(valor: float) -> str:
    """Convierte el ID de un agente a texto.

    Args:
        valor (float): ID del agente, o nulo.

    Returns:
        str: El ID sin decimales o "Sin agente" si era nulo.
    """
    if pd.isna(valor):
        return AGENT_MISSING_LABEL
    return str(int(valor))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena nulos, crea `tiene_empresa` y deja solo las columnas del modelo.

    Args:
        df (pd.DataFrame): DataFrame con las columnas originales del CSV.

    Returns:
        pd.DataFrame: Copia del DataFrame con solo las columnas de
        FEATURE_COLS.
    """
    df = df.copy()

    # children, nulo = 0 niños
    df["children"] = df["children"].fillna(CHILDREN_FILL_VALUE)

    # country, nulo = "Unknown"
    df["country"] = df["country"].fillna(COUNTRY_MISSING_LABEL)

    # agent, es un ID, lo paso a texto y el nulo pasa a ser "sin agente"
    df["agent"] = df["agent"].apply(agent_a_texto)

    # company, me quedo solo con si tiene empresa o no
    df[HAS_COMPANY_COL] = df[COMPANY_COL].notna().astype(int)

    # Quito las columnas de fuga de información y el ID de la empresa
    df = df.drop(columns=LEAKAGE_COLS + [COMPANY_COL], errors="ignore")

    return df[FEATURE_COLS].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los datos de entrenamiento.

    Quita duplicados y filas raras y deja solo las columnas del
    modelo más el objetivo.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpio con FEATURE_COLS y la columna objetivo.
    """
    filas_inicio = len(df)

    # Se eliminan duplicados
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {filas_inicio - len(df)}")

    df_limpio = prepare_features(df)
    df_limpio[TARGET] = df[TARGET]

    # Se eliminan filas raras, adr fuera de rango, sin huéspedes o de 0 noches
    huespedes = (
        df_limpio["adults"] + df_limpio["children"] + df_limpio["babies"]
    )
    noches = (
        df_limpio["stays_in_weekend_nights"]
        + df_limpio["stays_in_week_nights"]
    )
    filas_validas = (
        df_limpio["adr"].between(ADR_MIN, ADR_MAX)
        & (huespedes >= MIN_GUESTS)
        & (noches >= MIN_NIGHTS)
    )
    df_limpio = df_limpio[filas_validas].reset_index(drop=True)

    print(f"Filas raras eliminadas: {(~filas_validas).sum()}")
    print(f"Filas finales: {len(df_limpio)} de {filas_inicio}")

    return df_limpio


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos limpios en train y test.

    Mantiene la proporción de clases en los dos conjuntos.

    Args:
        df (pd.DataFrame): DataFrame limpio (salida de `clean_data`).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Una tupla
        con cuatro valores para entrenar y evaluar el modelo.
    """
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Construye el transformador de columnas, sin ajustar todavía.

    Returns:
        ColumnTransformer: Transformador con tres bloques: numéricas,
        binarias y categóricas.
    """
    # Numéricas: imputamos con la mediana y estandarizamos
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("binary", "passthrough", BINARY_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )
