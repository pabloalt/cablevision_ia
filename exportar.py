"""
exportar.py
Exportación de datos y resultados a CSV
Cable Visión IA — UPSJB 2026
"""

import os
import logging
import pandas as pd
from datetime import datetime
from config import EXPORTS_DIR

log = logging.getLogger(__name__)


def _ruta(nombre: str) -> str:
    fecha = datetime.now().strftime("%Y%m%d")
    archivo = f"{nombre}_{fecha}.csv"
    return os.path.join(EXPORTS_DIR, archivo)


def exportar_csv(df: pd.DataFrame, nombre: str,
                 encoding: str = "utf-8-sig") -> str:
    """Exporta cualquier DataFrame a CSV con timestamp en el nombre."""
    ruta = _ruta(nombre)
    df.to_csv(ruta, index=False, encoding=encoding)
    log.info(f"CSV exportado: {ruta} ({len(df)} filas)")
    print(f"✅ Exportado: {ruta}")
    return ruta


def exportar_dataset_churn(df: pd.DataFrame) -> str:
    """Exporta el dataset completo de perfiles para entrenamiento."""
    return exportar_csv(df, "perfil_clientes_churn")


def exportar_predicciones(df_pred: pd.DataFrame) -> str:
    """Exporta las predicciones de churn de clientes activos."""
    cols_export = [
        "ClienteID", "NombreCompleto", "EstadoCliente",
        "Plan", "MesesComoCliente", "ScoreChurn", "NivelRiesgo"
    ]
    cols = [c for c in cols_export if c in df_pred.columns]
    return exportar_csv(df_pred[cols], "predicciones_churn")


def exportar_clientes_alto_riesgo(df_pred: pd.DataFrame) -> str:
    """Exporta solo los clientes con riesgo Alto para el equipo comercial."""
    df_alto = df_pred[df_pred["NivelRiesgo"] == "Alto"].copy()
    log.info(f"Clientes de alto riesgo: {len(df_alto)}")
    return exportar_csv(df_alto, "clientes_alto_riesgo")


def exportar_segmentos(df_seg: pd.DataFrame) -> str:
    """Exporta la segmentación K-Means de clientes."""
    return exportar_csv(df_seg, "segmentos_clientes")


def exportar_anomalias(df_anom: pd.DataFrame) -> str:
    """Exporta las anomalías detectadas por Isolation Forest."""
    df_filtrado = df_anom[df_anom["EsAnomalia"] == True].copy()
    return exportar_csv(df_filtrado, "anomalias_fallas")


def exportar_stock_critico(df_stock: pd.DataFrame) -> str:
    """Exporta el inventario con stock crítico o bajo."""
    return exportar_csv(df_stock, "stock_critico")


def exportar_comparacion_modelos(df_comp: pd.DataFrame) -> str:
    """Exporta la tabla comparativa de métricas entre modelos."""
    return exportar_csv(df_comp.reset_index(), "comparacion_modelos")


def listar_exports() -> None:
    """Muestra todos los archivos exportados."""
    archivos = sorted(os.listdir(EXPORTS_DIR))
    if not archivos:
        print("No hay archivos exportados aún.")
        return
    print(f"\n📁 Archivos en {EXPORTS_DIR}:")
    for a in archivos:
        ruta = os.path.join(EXPORTS_DIR, a)
        size = os.path.getsize(ruta) / 1024
        print(f"   {a}  ({size:.1f} KB)")
