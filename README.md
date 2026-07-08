# Cable Visión IA — Proyecto Python

**Sistema de Inteligencia Artificial para optimizar la gestión y toma de decisiones**  
**UPSJB — Inteligencia Artificial 2026**

---

## Estructura del proyecto

```
cablevision_ia/
├── main.py                         ← Pipeline principal (ejecutar aquí)
├── config.py                       ← Configuración y parámetros
├── database.py                     ← Conexión SQL Server y consultas
├── preprocesamiento.py             ← Limpieza, features y split
├── modelos.py                      ← XGBoost, RF, Isolation Forest, K-Means
├── exportar.py                     ← Exportación a CSV
├── requirements.txt                ← Dependencias
├── models/                         ← Modelos entrenados (.pkl)
│   ├── churn_xgb_v1.pkl
│   ├── churn_rf_v2.pkl
│   ├── fallas_if_v1.pkl
│   └── segmento_km_v1.pkl
├── exports/                        ← Archivos CSV exportados
└── notebooks/
    └── analisis_exploratorio.py    ← Análisis visual con gráficos
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Configuración

Editar `config.py` y cambiar la contraseña:

```python
DB_CONFIG = {
    "server"  : "localhost",
    "database": "CableVisionIA",
    "uid"     : "sa",
    "pwd"     : "tu_contraseña",   # ← aquí
}
```

---

## Ejecución

### Pipeline completo (recomendado)
```bash
python main.py
```

Ejecuta en orden:
1. Churn (XGBoost + Random Forest)
2. Segmentación K-Means
3. Detección de fallas (Isolation Forest)
4. Stock crítico

### Solo predicción de churn
```bash
python main.py --solo-churn
```

### Predecir con modelo ya entrenado
```bash
python main.py --solo-prediccion
```

### Solo segmentación
```bash
python main.py --solo-segmentos
```

### Solo detección de fallas
```bash
python main.py --solo-fallas
```

### Solo exportar datos a CSV
```bash
python main.py --exportar
```

---

## Modelos entrenados

| Archivo | Modelo | Caso de uso |
|---------|--------|-------------|
| `churn_xgb_v1.pkl` | XGBoost | Predicción de churn |
| `churn_rf_v2.pkl` | Random Forest | Churn (mejor recall) |
| `fallas_if_v1.pkl` | Isolation Forest | Detección de anomalías |
| `segmento_km_v1.pkl` | K-Means (k=4) | Segmentación de clientes |

---

## Archivos CSV generados

| Archivo | Descripción |
|---------|-------------|
| `perfil_clientes_churn_YYYYMMDD.csv` | Dataset completo de features |
| `predicciones_churn_YYYYMMDD.csv` | Score de churn por cliente activo |
| `clientes_alto_riesgo_YYYYMMDD.csv` | Solo clientes con riesgo Alto |
| `segmentos_clientes_YYYYMMDD.csv` | Cluster asignado por cliente |
| `anomalias_fallas_YYYYMMDD.csv` | Zonas con comportamiento anómalo |
| `stock_critico_YYYYMMDD.csv` | Equipos con stock bajo o crítico |
| `comparacion_modelos_YYYYMMDD.csv` | Métricas XGBoost vs Random Forest |

---

## Features del modelo de churn

| Feature | Descripción |
|---------|-------------|
| MesesComoCliente | Antigüedad del cliente |
| TotalPagos | Total de recibos generados |
| PagosPuntuales | Pagos realizados a tiempo |
| PagosVencidos | Pagos no realizados |
| PromedioAtraso | Días promedio de retraso |
| TotalIncidencias | Reportes técnicos del cliente |
| PromedioResolucionMin | Tiempo promedio de resolución |
| PromedioCalificacion | Satisfacción promedio (1-5) |
| TotalAtenciones | Contactos al soporte |
| PrecioMensual | Plan contratado (S/) |
| TasaPuntualidad | PagosPuntuales / TotalPagos |
| IncidenciasPorMes | Incidencias / MesesComoCliente |
| IndSatisfaccion | Calificacion × TasaPuntualidad |
| TieneDeuda | 1 si tiene pagos vencidos |
| MultiplesIncidencias | 1 si tiene 3+ incidencias |
| AtrasoPonderado | PromedioAtraso × PagosVencidos |
| **Churn** | **Variable objetivo: 0=No / 1=Sí** |
