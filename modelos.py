"""
modelos.py
Entrenamiento, evaluación y persistencia de todos los modelos IA
Cable Visión IA — UPSJB 2026

Modelos incluidos:
  1. XGBoost      — Predicción de churn
  2. Random Forest — Predicción de churn (alternativo)
  3. Isolation Forest — Detección de anomalías / fallas
  4. K-Means      — Segmentación de clientes
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    silhouette_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import (
    MODELS_DIR, XGBOOST_PARAMS, CHURN_FEATURES,
    KMEANS_N_CLUSTERS, KMEANS_NOMBRES_SEGMENTOS,
    CHURN_THRESHOLD_ALTO, CHURN_THRESHOLD_MEDIO,
    MLFLOW_EXPERIMENT
)

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# MODELO 1 — XGBoost (Predicción de Churn)
# ════════════════════════════════════════════════════════════════

def entrenar_xgboost(X_train, y_train, X_val, y_val,
                     optimizar: bool = False):
    """
    Entrena el modelo XGBoost para predicción de churn.

    Args:
        X_train, y_train : datos de entrenamiento
        X_val, y_val     : datos de validación
        optimizar        : si True, ejecuta GridSearchCV

    Returns:
        modelo entrenado
    """
    log.info("Entrenando modelo XGBoost para predicción de churn...")

    if optimizar:
        log.info("Optimizando hiperparámetros con GridSearchCV...")
        param_grid = {
            "n_estimators"  : [100, 200, 300],
            "max_depth"     : [3, 5, 7],
            "learning_rate" : [0.01, 0.05, 0.10],
            "subsample"     : [0.7, 0.8, 1.0],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base = XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        grid = GridSearchCV(base, param_grid, cv=cv,
                            scoring="roc_auc", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_
        log.info(f"Mejores hiperparámetros: {grid.best_params_}")
    else:
        modelo = XGBClassifier(**XGBOOST_PARAMS)
        modelo.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Evaluación en validación
    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)[:, 1]
    metricas = evaluar_clasificador(y_val, y_pred, y_prob, nombre="XGBoost")

    # Guardar modelo
    ruta = os.path.join(MODELS_DIR, "churn_xgb_v1.pkl")
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    log.info(f"Modelo guardado en: {ruta}")

    return modelo, metricas, ruta


# ════════════════════════════════════════════════════════════════
# MODELO 2 — Random Forest (Predicción de Churn alternativo)
# ════════════════════════════════════════════════════════════════

def entrenar_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un Random Forest como modelo alternativo de churn.
    Suele tener mejor recall que XGBoost en datasets pequeños.
    """
    log.info("Entrenando modelo Random Forest...")

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # compensa desbalance sin SMOTE
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_val)
    y_prob = modelo.predict_proba(X_val)[:, 1]
    metricas = evaluar_clasificador(y_val, y_pred, y_prob,
                                    nombre="Random Forest")

    ruta = os.path.join(MODELS_DIR, "churn_rf_v2.pkl")
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    log.info(f"Modelo guardado en: {ruta}")

    return modelo, metricas, ruta


# ════════════════════════════════════════════════════════════════
# MODELO 3 — Isolation Forest (Detección de anomalías / fallas)
# ════════════════════════════════════════════════════════════════

def entrenar_isolation_forest(df_incidencias: pd.DataFrame):
    """
    Entrena Isolation Forest sobre el historial de incidencias
    por zona para detectar patrones anómalos (posibles fallas masivas).

    Args:
        df_incidencias: resultado de vw_IncidenciasPorZona

    Returns:
        modelo, DataFrame con score de anomalía por zona
    """
    log.info("Entrenando Isolation Forest para detección de fallas...")

    features_if = [
        "TotalIncidencias",
        "PromedioTiempoAtencionMin",
        "PromedioTiempoResolucionMin",
        "PromedioCalificacion",
    ]

    df = df_incidencias[features_if].fillna(0).copy()

    modelo = IsolationForest(
        n_estimators=200,
        contamination=0.10,   # esperamos ~10% de observaciones anómalas
        random_state=42
    )
    modelo.fit(df)

    # Score de anomalía: valores más negativos = más anómalos
    scores = modelo.decision_function(df)
    # Normalizar a 0–1 (1 = más anómalo)
    score_norm = 1 - (scores - scores.min()) / (scores.max() - scores.min())

    df_resultado = df_incidencias[["Zona", "Periodo",
                                   "Categoria", "TotalIncidencias"]].copy()
    df_resultado["ScoreAnomalia"] = score_norm
    df_resultado["EsAnomalia"]    = modelo.predict(df) == -1
    df_resultado = df_resultado.sort_values("ScoreAnomalia", ascending=False)

    log.info(f"Anomalías detectadas: {df_resultado['EsAnomalia'].sum()} "
             f"de {len(df_resultado)} observaciones")

    ruta = os.path.join(MODELS_DIR, "fallas_if_v1.pkl")
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    log.info(f"Modelo guardado en: {ruta}")

    return modelo, df_resultado, ruta


# ════════════════════════════════════════════════════════════════
# MODELO 4 — K-Means (Segmentación de clientes)
# ════════════════════════════════════════════════════════════════

def entrenar_kmeans(df_clientes: pd.DataFrame,
                    n_clusters: int = KMEANS_N_CLUSTERS):
    """
    Segmenta los clientes en grupos usando K-Means.

    Returns:
        modelo, DataFrame con ClienteID, ClusterID, NombreSegmento
    """
    log.info(f"Entrenando K-Means con k={n_clusters}...")

    features_km = [
        "MesesComoCliente",
        "TasaPuntualidad",
        "PromedioAtraso",
        "TotalIncidencias",
        "PromedioCalificacion",
        "PrecioMensual",
        "TotalAtenciones",
    ]

    # Solo columnas disponibles
    cols = [c for c in features_km if c in df_clientes.columns]
    X = df_clientes[cols].fillna(0).copy()

    # Elegir k óptimo con método del codo
    inercias = []
    silhouettes = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    log.info(f"Silhouette scores: "
             + ", ".join(f"k={k}:{s:.3f}" for k,s in zip(k_range, silhouettes)))

    # Entrenar con k elegido
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = modelo.fit_predict(X)

    sil = silhouette_score(X, labels)
    log.info(f"Silhouette score final (k={n_clusters}): {sil:.4f}")

    # Construir resultado
    df_seg = df_clientes[["ClienteID"]].copy()
    df_seg["ClusterID"]      = labels
    df_seg["NombreSegmento"] = df_seg["ClusterID"].map(KMEANS_NOMBRES_SEGMENTOS)

    log.info("Distribución de segmentos:")
    for seg, cnt in df_seg["NombreSegmento"].value_counts().items():
        log.info(f"  {seg}: {cnt} clientes")

    ruta = os.path.join(MODELS_DIR, "segmento_km_v1.pkl")
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    log.info(f"Modelo guardado en: {ruta}")

    return modelo, df_seg, ruta, sil


# ════════════════════════════════════════════════════════════════
# EVALUACIÓN DE CLASIFICADORES
# ════════════════════════════════════════════════════════════════

def evaluar_clasificador(y_true, y_pred, y_prob,
                         nombre: str = "Modelo") -> dict:
    """
    Calcula y muestra las métricas principales de un clasificador.
    Retorna diccionario con todas las métricas.
    """
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  EVALUACIÓN — {nombre}")
    print(f"{'='*50}")
    print(f"  Precisión : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Matriz de confusión:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Churn','Churn'])}")

    return {
        "nombre"   : nombre,
        "precision": round(prec, 4),
        "recall"   : round(rec, 4),
        "f1"       : round(f1, 4),
        "auc_roc"  : round(auc, 4),
    }


def comparar_modelos(metricas_list: list) -> pd.DataFrame:
    """Muestra tabla comparativa de métricas entre modelos."""
    df = pd.DataFrame(metricas_list)
    df = df.set_index("nombre")
    print("\n=== COMPARACIÓN DE MODELOS ===")
    print(df.to_string())
    return df


# ════════════════════════════════════════════════════════════════
# INFERENCIA — Predicción sobre nuevos datos
# ════════════════════════════════════════════════════════════════

def predecir_churn(modelo, df_activos: pd.DataFrame,
                   features: list = None) -> pd.DataFrame:
    """
    Genera predicciones de churn sobre clientes activos.

    Returns:
        DataFrame con ClienteID, NombreCompleto, ScoreChurn, NivelRiesgo
    """
    if features is None:
        from preprocesamiento import FEATURES_EXTENDIDAS
        features = FEATURES_EXTENDIDAS

    cols = [c for c in features if c in df_activos.columns]
    X    = df_activos[cols].fillna(0)

    scores = modelo.predict_proba(X)[:, 1]

    cols_pred = ["ClienteID", "NombreCompleto", "EstadoCliente", "MesesComoCliente"]
    if "Plan" in df_activos.columns:
        cols_pred.insert(3, "Plan")
    elif "PlanD" in df_activos.columns:
        cols_pred.insert(3, "PlanD")
    df_pred = df_activos[cols_pred].copy()
    df_pred["ScoreChurn"]  = scores.round(4)
    df_pred["NivelRiesgo"] = pd.cut(
        df_pred["ScoreChurn"],
        bins=[-0.001, CHURN_THRESHOLD_MEDIO,
              CHURN_THRESHOLD_ALTO, 1.001],
        labels=["Bajo", "Medio", "Alto"]
    )
    df_pred = df_pred.sort_values("ScoreChurn", ascending=False)

    log.info(f"Predicciones generadas para {len(df_pred)} clientes")
    log.info(f"  Alto:  {(df_pred['NivelRiesgo']=='Alto').sum()}")
    log.info(f"  Medio: {(df_pred['NivelRiesgo']=='Medio').sum()}")
    log.info(f"  Bajo:  {(df_pred['NivelRiesgo']=='Bajo').sum()}")

    return df_pred


def cargar_modelo(nombre_archivo: str):
    """Carga un modelo serializado desde la carpeta models/."""
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    with open(ruta, "rb") as f:
        modelo = pickle.load(f)
    log.info(f"Modelo cargado desde: {ruta}")
    return modelo
