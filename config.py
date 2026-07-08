"""
config.py
Configuración de conexión a SQL Server y parámetros del proyecto
Cable Visión IA — UPSJB 2026
"""

import os

# ── Conexión SQL Server ───────────────────────────────────────
DB_CONFIG = {
    "driver"  : "ODBC Driver 17 for SQL Server",
    "server"  : os.getenv("DB_SERVER",   "181.119.93.26,1433"),
    "database": os.getenv("DB_NAME",     "CableVisionIA"),
    "uid"     : os.getenv("DB_USER",     "sa"),
    "pwd"     : os.getenv("DB_PASSWORD", "@OmnisatSql@"),
}

CONNECTION_STRING = (
    f"DRIVER={{{DB_CONFIG['driver']}}};"
    f"SERVER={DB_CONFIG['server']};"
    f"DATABASE={DB_CONFIG['database']};"
    f"UID={DB_CONFIG['uid']};"
    f"PWD={DB_CONFIG['pwd']}"
)

# ── Rutas del proyecto ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

for d in [MODELS_DIR, EXPORTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Parámetros del modelo de churn ───────────────────────────
CHURN_FEATURES = [
    "MesesComoCliente",
    "TotalPagos",
    "PagosPuntuales",
    "PagosVencidos",
    "PromedioAtraso",
    "TotalIncidencias",
    "PromedioResolucionMin",
    "PromedioCalificacion",
    "TotalAtenciones",
    "PrecioMensual",
]

CHURN_TARGET    = "Churn"
CHURN_THRESHOLD_ALTO  = 0.65
CHURN_THRESHOLD_MEDIO = 0.40

# ── Parámetros XGBoost ────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"      : 200,
    "max_depth"         : 5,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "eval_metric"       : "logloss",
    "random_state"      : 42,
    "use_label_encoder" : False,
}

# ── Parámetros K-Means ────────────────────────────────────────
KMEANS_N_CLUSTERS = 4
KMEANS_NOMBRES_SEGMENTOS = {
    0: "Alto valor",
    1: "Riesgo abandono",
    2: "Demanda estacional",
    3: "Nuevo sin historial",
}

# ── MLflow ────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "CableVision_Churn"
