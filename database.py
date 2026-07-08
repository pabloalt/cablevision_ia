"""
database.py
Módulo de conexión y consultas a SQL Server
Cable Visión IA — UPSJB 2026
"""

import pyodbc
import pandas as pd
import logging
from config import CONNECTION_STRING

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Conexión ──────────────────────────────────────────────────
def get_connection():
    """Retorna una conexión activa a SQL Server."""
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        log.info("Conexión a SQL Server establecida.")
        return conn
    except Exception as e:
        log.error(f"Error al conectar a SQL Server: {e}")
        raise


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta una consulta SELECT y retorna un DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
        log.info(f"Query ejecutada: {len(df)} filas retornadas.")
        return df
    finally:
        conn.close()


def execute(sql: str, params=None, many=False):
    """Ejecuta INSERT / UPDATE / DELETE."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if many and params:
            cur.executemany(sql, params)
        elif params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        conn.commit()
        log.info(f"Sentencia ejecutada. Filas afectadas: {cur.rowcount}")
    finally:
        conn.close()


# ── Cargar datasets ───────────────────────────────────────────
def cargar_perfil_churn() -> pd.DataFrame:
    """Carga el dataset completo de perfiles de clientes para churn."""
    sql = """
        SELECT *
        FROM vw_PerfilClienteChurn
        ORDER BY ClienteID
    """
    df = query_df(sql)
    log.info(f"Dataset churn: {len(df)} clientes | "
             f"Churn=1: {df['Churn'].sum()} | "
             f"Churn=0: {(df['Churn']==0).sum()}")
    return df


def cargar_demanda_instalaciones() -> pd.DataFrame:
    """Carga la demanda histórica de instalaciones por zona."""
    return query_df("""
        SELECT *
        FROM vw_DemandaInstalacionesMensual
        ORDER BY Zona, Periodo
    """)


def cargar_incidencias_zona() -> pd.DataFrame:
    """Carga el historial de incidencias para detección de anomalías."""
    return query_df("""
        SELECT *
        FROM vw_IncidenciasPorZona
        ORDER BY TotalIncidencias DESC
    """)


def cargar_stock_critico() -> pd.DataFrame:
    """Retorna equipos con stock bajo o crítico."""
    return query_df("""
        SELECT *
        FROM vw_AlertaStockCritico
        WHERE EstadoStock IN ('CRITICO','BAJO')
        ORDER BY DiferenciaMinimo ASC
    """)


def cargar_clientes_activos() -> pd.DataFrame:
    """Carga solo clientes activos para predicción."""
    return query_df("""
        SELECT *
        FROM vw_PerfilClienteChurn
        WHERE EstadoCliente = 'Activo'
        ORDER BY ClienteID
    """)


# ── Guardar predicciones ──────────────────────────────────────
def guardar_predicciones_churn(df_pred: pd.DataFrame, modelo_id: int = 1):
    """
    Inserta las predicciones de churn en la tabla PrediccionesChurn.
    df_pred debe tener columnas: ClienteID, ScoreChurn, NivelRiesgo
    """
    rows = [
        (modelo_id,
         int(row["ClienteID"]),
         float(row["ScoreChurn"]),
         row["NivelRiesgo"])
        for _, row in df_pred.iterrows()
    ]
    sql = """
        INSERT INTO PrediccionesChurn
            (ModeloID, ClienteID, ScoreChurn, NivelRiesgo)
        VALUES (?, ?, ?, ?)
    """
    execute(sql, params=rows, many=True)
    log.info(f"{len(rows)} predicciones guardadas en PrediccionesChurn.")


def guardar_segmentos(df_seg: pd.DataFrame, modelo_id: int = 5):
    """Inserta los segmentos K-Means en la tabla SegmentosCliente."""
    rows = [
        (modelo_id,
         int(row["ClienteID"]),
         int(row["ClusterID"]),
         row["NombreSegmento"])
        for _, row in df_seg.iterrows()
    ]
    sql = """
        INSERT INTO SegmentosCliente
            (ModeloID, ClienteID, ClusterID, NombreSegmento)
        VALUES (?, ?, ?, ?)
    """
    execute(sql, params=rows, many=True)
    log.info(f"{len(rows)} segmentos guardados en SegmentosCliente.")


def registrar_modelo(nombre, version, algoritmo, caso_uso,
                     precision, recall, f1, auc_roc, ruta):
    """Registra un nuevo modelo entrenado en la tabla ModelosIA."""
    sql = """
        INSERT INTO ModelosIA
            (Nombre, Version, Algoritmo, CasoDeUso,
             Precision, Recall, F1Score, AUC_ROC, RutaModelo, Activo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """
    execute(sql, params=(nombre, version, algoritmo, caso_uso,
                         precision, recall, f1, auc_roc, ruta))
    log.info(f"Modelo '{nombre} v{version}' registrado en ModelosIA.")
