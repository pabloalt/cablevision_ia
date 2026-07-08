"""
preprocesamiento.py
Limpieza, transformación y preparación de features
Cable Visión IA — UPSJB 2026
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from config import CHURN_FEATURES, CHURN_TARGET

log = logging.getLogger(__name__)


# ── Limpieza general ──────────────────────────────────────────
def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza básica al dataset:
    - Rellena nulos numéricos con 0
    - Rellena nulos de calificación con la mediana
    - Elimina duplicados por ClienteID
    """
    df = df.copy()

    # Eliminar duplicados
    antes = len(df)
    df = df.drop_duplicates(subset=["ClienteID"])
    log.info(f"Duplicados eliminados: {antes - len(df)}")

    # Calificación: usar mediana si es nulo (cliente sin incidencias)
    if "PromedioCalificacion" in df.columns:
        mediana = df["PromedioCalificacion"].median()
        df["PromedioCalificacion"] = df["PromedioCalificacion"].fillna(mediana)

    # Tiempo de resolución: 0 si no tuvo incidencias
    if "PromedioResolucionMin" in df.columns:
        df["PromedioResolucionMin"] = df["PromedioResolucionMin"].fillna(0)

    # Resto de nulos numéricos → 0
    df = df.fillna(0)

    log.info(f"Dataset limpio: {len(df)} filas, {df.shape[1]} columnas")
    return df


# ── Feature engineering ───────────────────────────────────────
def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas que mejoran la predicción del modelo.
    """
    df = df.copy()

    # Tasa de pagos puntuales (0–1)
    df["TasaPuntualidad"] = np.where(
        df["TotalPagos"] > 0,
        df["PagosPuntuales"] / df["TotalPagos"],
        0
    )

    # Tasa de incidencias por mes de permanencia
    df["IncidenciasPorMes"] = np.where(
        df["MesesComoCliente"] > 0,
        df["TotalIncidencias"] / df["MesesComoCliente"],
        0
    )

    # Índice de satisfacción combinado (calificación × tasa puntualidad)
    df["IndSatisfaccion"] = (
        df["PromedioCalificacion"] / 5.0 * df["TasaPuntualidad"]
    )

    # Flag: cliente con deuda (al menos 1 pago vencido)
    df["TieneDeuda"] = (df["PagosVencidos"] > 0).astype(int)

    # Flag: cliente con múltiples incidencias graves
    df["MultiplesIncidencias"] = (df["TotalIncidencias"] >= 3).astype(int)

    # Atraso promedio ponderado
    df["AtrasoPonderado"] = df["PromedioAtraso"] * df["PagosVencidos"]

    log.info(f"Features creadas: TasaPuntualidad, IncidenciasPorMes, "
             f"IndSatisfaccion, TieneDeuda, MultiplesIncidencias, AtrasoPonderado")
    return df


# ── Features extendidas ───────────────────────────────────────
FEATURES_EXTENDIDAS = CHURN_FEATURES + [
    "TasaPuntualidad",
    "IncidenciasPorMes",
    "IndSatisfaccion",
    "TieneDeuda",
    "MultiplesIncidencias",
    "AtrasoPonderado",
]


# ── Split train/val/test ──────────────────────────────────────
def dividir_dataset(df: pd.DataFrame,
                    features: list = None,
                    test_size: float = 0.15,
                    val_size: float = 0.15,
                    random_state: int = 42):
    """
    Divide el dataset en train (70%) / validación (15%) / test (15%).
    Retorna: X_train, X_val, X_test, y_train, y_val, y_test
    """
    if features is None:
        features = FEATURES_EXTENDIDAS

    X = df[features].copy()
    y = df[CHURN_TARGET].copy()

    # Primer split: separar test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Segundo split: separar validación del resto
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )

    log.info(f"División del dataset:")
    log.info(f"  Train:      {len(X_train)} registros ({len(X_train)/len(df)*100:.0f}%)")
    log.info(f"  Validación: {len(X_val)} registros ({len(X_val)/len(df)*100:.0f}%)")
    log.info(f"  Test:       {len(X_test)} registros ({len(X_test)/len(df)*100:.0f}%)")
    log.info(f"  Churn en train: {y_train.sum()}/{len(y_train)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Balanceo con SMOTE ────────────────────────────────────────
def balancear_clases(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     random_state: int = 42):
    """
    Aplica SMOTE para balancear las clases cuando hay
    desbalance entre clientes churn (1) y no-churn (0).
    """
    conteo = y_train.value_counts()
    ratio  = conteo.min() / conteo.max()
    log.info(f"Balance antes de SMOTE: {dict(conteo)} (ratio={ratio:.2f})")

    if ratio < 0.40:
        smote = SMOTE(random_state=random_state)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        log.info(f"SMOTE aplicado. Nuevo balance: {dict(pd.Series(y_bal).value_counts())}")
        return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal)
    else:
        log.info("Dataset suficientemente balanceado. SMOTE no aplicado.")
        return X_train, y_train


# ── Escalado ──────────────────────────────────────────────────
def escalar_features(X_train, X_val, X_test, metodo="standard"):
    """
    Escala las features. Ajusta el scaler solo con train.
    metodo: 'standard' (media=0, std=1) o 'minmax' (0–1)
    """
    Scaler = StandardScaler if metodo == "standard" else MinMaxScaler
    scaler = Scaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    log.info(f"Escalado aplicado ({metodo}). "
             f"Media train: {X_train_sc.mean().mean():.4f}")
    return X_train_sc, X_val_sc, X_test_sc, scaler
