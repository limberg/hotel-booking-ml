from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset_practica_final.csv"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_TESTS_DIR = MODELS_DIR / "tests"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
RESULTS_TABLE_PATH = METRICS_DIR / "resultados_modelos.csv"

# Aleatoriedad y partición de los datos
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Variable dependiente
TARGET = "is_canceled"

# Fuga de información
LEAKAGE_COLS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
]

# Limpieza de filas
ADR_MIN = 0
ADR_MAX = 1000
MIN_GUESTS = 1
MIN_NIGHTS = 1

# Nulos y variables nuevas
CHILDREN_FILL_VALUE = 0
COUNTRY_MISSING_LABEL = "Unknown"
AGENT_MISSING_LABEL = "Sin agente"

COMPANY_COL = "company"
HAS_COMPANY_COL = "tiene_empresa"

# Columnas que usará el modelo
NUMERIC_COLS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

BINARY_COLS = [
    "is_repeated_guest",
    HAS_COMPANY_COL,
]

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]

FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

# Preprocesado
ONEHOT_MIN_FREQUENCY = 100
