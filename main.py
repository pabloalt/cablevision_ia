"""
main.py
Pipeline principal del sistema IA de Cable Visión
Ejecuta en orden: carga → limpieza → entrenamiento → predicción → exportación
Cable Visión IA — UPSJB 2026

Uso:
    python main.py                    # pipeline completo
    python main.py --solo-churn       # solo modelo de churn
    python main.py --solo-prediccion  # solo predice con modelo guardado
    python main.py --exportar         # solo exporta a CSV
"""

import argparse
import logging
import sys
from datetime import datetime

# Módulos propios
import database    as db
import preprocesamiento as prep
import modelos     as mod
import exportar    as exp
from config import CHURN_FEATURES
from preprocesamiento import FEATURES_EXTENDIDAS

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
    ]
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════

def pipeline_churn():
    """
    Pipeline completo del modelo de predicción de churn:
    1. Cargar datos desde SQL Server
    2. Limpiar y crear features
    3. Dividir en train/val/test
    4. Balancear con SMOTE
    5. Entrenar XGBoost y Random Forest
    6. Comparar modelos
    7. Guardar predicciones en BD
    8. Exportar a CSV
    """
    print("\n" + "="*60)
    print("  PIPELINE — PREDICCIÓN DE CHURN")
    print("="*60)

    # ── 1. Cargar datos ───────────────────────────────────────
    log.info("Paso 1: Cargando datos desde SQL Server...")
    df = db.cargar_perfil_churn()
    print(f"\n  Total clientes: {len(df)}")
    print(f"  Churn = 1 (cancelados): {df['Churn'].sum()}")
    print(f"  Churn = 0 (activos):    {(df['Churn']==0).sum()}")
    print(f"  Columnas disponibles:   {list(df.columns)}\n")

    # ── 2. Limpiar y enriquecer ───────────────────────────────
    log.info("Paso 2: Limpieza y feature engineering...")
    df = prep.limpiar_dataset(df)
    df = prep.crear_features(df)

    # Exportar dataset limpio
    exp.exportar_dataset_churn(df)

    # ── 3. Dividir ────────────────────────────────────────────
    log.info("Paso 3: Dividiendo dataset 70/15/15...")
    X_train, X_val, X_test, y_train, y_val, y_test = prep.dividir_dataset(df)

    # ── 4. Balancear ──────────────────────────────────────────
    log.info("Paso 4: Balanceando clases con SMOTE...")
    X_train_b, y_train_b = prep.balancear_clases(X_train, y_train)

    # ── 5. Entrenar modelos ───────────────────────────────────
    log.info("Paso 5a: Entrenando XGBoost...")
    modelo_xgb, met_xgb, ruta_xgb = mod.entrenar_xgboost(
        X_train_b, y_train_b, X_val, y_val
    )

    log.info("Paso 5b: Entrenando Random Forest...")
    modelo_rf, met_rf, ruta_rf = mod.entrenar_random_forest(
        X_train_b, y_train_b, X_val, y_val
    )

    # ── 6. Comparar en test ───────────────────────────────────
    log.info("Paso 6: Evaluación final en conjunto de test...")
    print("\n📊 EVALUACIÓN EN TEST SET:")

    y_pred_xgb = modelo_xgb.predict(X_test)
    y_prob_xgb = modelo_xgb.predict_proba(X_test)[:, 1]
    met_xgb_test = mod.evaluar_clasificador(y_test, y_pred_xgb, y_prob_xgb, "XGBoost [TEST]")

    y_pred_rf  = modelo_rf.predict(X_test)
    y_prob_rf  = modelo_rf.predict_proba(X_test)[:, 1]
    met_rf_test = mod.evaluar_clasificador(y_test, y_pred_rf, y_prob_rf, "Random Forest [TEST]")

    df_comp = mod.comparar_modelos([met_xgb_test, met_rf_test])
    exp.exportar_comparacion_modelos(df_comp)

    # Registrar el mejor modelo (XGBoost) en la BD
    db.registrar_modelo(
        nombre    = "Predictor Churn XGBoost",
        version   = "1.1",
        algoritmo = "XGBoost",
        caso_uso  = "Churn",
        precision = met_xgb_test["precision"],
        recall    = met_xgb_test["recall"],
        f1        = met_xgb_test["f1"],
        auc_roc   = met_xgb_test["auc_roc"],
        ruta      = ruta_xgb
    )

    # ── 7. Predecir sobre clientes activos ────────────────────
    log.info("Paso 7: Generando predicciones sobre clientes activos...")
    df_activos = db.cargar_clientes_activos()
    df_activos = prep.limpiar_dataset(df_activos)
    df_activos = prep.crear_features(df_activos)

    df_pred = mod.predecir_churn(modelo_xgb, df_activos)

    # Guardar en BD y exportar
    db.guardar_predicciones_churn(df_pred[["ClienteID", "ScoreChurn", "NivelRiesgo"]])
    exp.exportar_predicciones(df_pred)
    exp.exportar_clientes_alto_riesgo(df_pred)

    print(f"\n✅ Pipeline de churn completado.")
    print(f"   Clientes en riesgo ALTO: "
          f"{(df_pred['NivelRiesgo']=='Alto').sum()}")
    print(f"   Ver resultados en: exports/")
    return modelo_xgb, df_pred


def pipeline_segmentacion():
    """
    Pipeline de segmentación K-Means:
    1. Cargar clientes activos
    2. Limpiar y crear features
    3. Entrenar K-Means
    4. Guardar segmentos en BD
    5. Exportar CSV
    """
    print("\n" + "="*60)
    print("  PIPELINE — SEGMENTACIÓN DE CLIENTES (K-Means)")
    print("="*60)

    df = db.cargar_clientes_activos()
    df = prep.limpiar_dataset(df)
    df = prep.crear_features(df)

    modelo_km, df_seg, ruta_km, silhouette = mod.entrenar_kmeans(df)

    db.guardar_segmentos(df_seg)
    exp.exportar_segmentos(df_seg)

    print(f"\n✅ Segmentación completada. Silhouette = {silhouette:.4f}")
    print(df_seg["NombreSegmento"].value_counts().to_string())
    return modelo_km, df_seg


def pipeline_deteccion_fallas():
    """
    Pipeline Isolation Forest para detección de zonas con
    comportamiento anómalo en incidencias técnicas.
    """
    print("\n" + "="*60)
    print("  PIPELINE — DETECCIÓN DE FALLAS (Isolation Forest)")
    print("="*60)

    df_inc = db.cargar_incidencias_zona()
    if df_inc.empty:
        print("⚠️  Sin datos de incidencias por zona. Omitiendo Isolation Forest.")
        return None, None
    modelo_if, df_anom, ruta_if = mod.entrenar_isolation_forest(df_inc)

    exp.exportar_anomalias(df_anom)

    n_anom = df_anom["EsAnomalia"].sum()
    print(f"\n✅ Detección completada. Anomalías encontradas: {n_anom}")
    print(df_anom[df_anom["EsAnomalia"]].head(10).to_string(index=False))
    return modelo_if, df_anom


def pipeline_inventario():
    """Consulta el stock crítico y exporta recomendaciones."""
    print("\n" + "="*60)
    print("  STOCK CRÍTICO — CABLE VISIÓN")
    print("="*60)

    df_stock = db.cargar_stock_critico()
    exp.exportar_stock_critico(df_stock)

    print(f"\n✅ {len(df_stock)} zonas con stock bajo o crítico")
    print(df_stock.to_string(index=False))
    return df_stock


# ════════════════════════════════════════════════════════════════
# EJECUCIÓN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline IA Cable Visión — UPSJB 2026"
    )
    parser.add_argument("--solo-churn",       action="store_true",
                        help="Solo ejecuta el pipeline de churn")
    parser.add_argument("--solo-prediccion",  action="store_true",
                        help="Solo genera predicciones con modelo guardado")
    parser.add_argument("--solo-segmentos",   action="store_true",
                        help="Solo ejecuta K-Means")
    parser.add_argument("--solo-fallas",      action="store_true",
                        help="Solo ejecuta Isolation Forest")
    parser.add_argument("--exportar",         action="store_true",
                        help="Solo exporta datos actuales a CSV")
    args = parser.parse_args()

    inicio = datetime.now()
    print(f"\n🚀 Cable Visión IA — Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.solo_prediccion:
        # Cargar modelo guardado y generar predicciones
        modelo = mod.cargar_modelo("churn_xgb_v1.pkl")
        df_activos = db.cargar_clientes_activos()
        df_activos = prep.limpiar_dataset(df_activos)
        df_activos = prep.crear_features(df_activos)
        df_pred    = mod.predecir_churn(modelo, df_activos)
        exp.exportar_predicciones(df_pred)
        exp.exportar_clientes_alto_riesgo(df_pred)

    elif args.solo_churn:
        pipeline_churn()

    elif args.solo_segmentos:
        pipeline_segmentacion()

    elif args.solo_fallas:
        pipeline_deteccion_fallas()

    elif args.exportar:
        df = db.cargar_perfil_churn()
        exp.exportar_dataset_churn(df)
        exp.listar_exports()

    else:
        # Pipeline completo
        pipeline_churn()
        pipeline_segmentacion()
        pipeline_deteccion_fallas()
        pipeline_inventario()
        exp.listar_exports()

    fin = datetime.now()
    duracion = (fin - inicio).seconds
    print(f"\n✅ Pipeline finalizado en {duracion} segundos.")
    print(f"   Resultados en: exports/")
    print(f"   Modelos en:    models/")
    print(f"   Logs en:       logs/\n")


if __name__ == "__main__":
    main()
