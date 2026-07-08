# ============================================================
# analisis_exploratorio.py
# Análisis visual del dataset de Cable Visión
# Ejecutar celda por celda en Jupyter Notebook o como script
# Cable Visión IA — UPSJB 2026
# ============================================================

import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Estilo de gráficos
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f9fa",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
})
AZUL  = "#1A3F6F"
ROJO  = "#DC2626"
VERDE = "#16A34A"
GRIS  = "#94A3B8"


# ── Cargar datos ──────────────────────────────────────────────
import database as db
import preprocesamiento as prep

df_raw = db.cargar_perfil_churn()
df     = prep.limpiar_dataset(df_raw)
df     = prep.crear_features(df)

print(f"Dataset cargado: {df.shape}")
print(df.head())
print(df.describe().T)


# ═════════════════════════════════════════════════════════════
# GRÁFICO 1: Distribución de Churn
# ═════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribución de Churn — Cable Visión", y=1.02)

# Conteo
conteo = df["Churn"].value_counts()
axes[0].bar(["No Churn (0)", "Churn (1)"],
            conteo.values,
            color=[VERDE, ROJO], edgecolor="white", width=0.5)
axes[0].set_title("Conteo por clase")
axes[0].set_ylabel("Cantidad de clientes")
for i, v in enumerate(conteo.values):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# Porcentaje
axes[1].pie(conteo.values, labels=["No Churn", "Churn"],
            colors=[VERDE, ROJO], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 12})
axes[1].set_title("Proporción churn vs no churn")

plt.tight_layout()
plt.savefig("exports/01_distribucion_churn.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico 1 guardado.")


# ═════════════════════════════════════════════════════════════
# GRÁFICO 2: Distribución de features numéricas
# ═════════════════════════════════════════════════════════════
features_plot = [
    "MesesComoCliente", "PagosPuntuales", "PagosVencidos",
    "PromedioAtraso", "TotalIncidencias", "PromedioCalificacion",
    "TasaPuntualidad", "PrecioMensual"
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Distribución de features — por clase Churn")
axes = axes.flatten()

for i, feat in enumerate(features_plot):
    if feat not in df.columns:
        continue
    for churn_val, color, label in [(0, VERDE, "No Churn"), (1, ROJO, "Churn")]:
        datos = df[df["Churn"] == churn_val][feat].dropna()
        axes[i].hist(datos, bins=15, alpha=0.6, color=color, label=label)
    axes[i].set_title(feat)
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("exports/02_distribucion_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico 2 guardado.")


# ═════════════════════════════════════════════════════════════
# GRÁFICO 3: Correlación con la variable Churn
# ═════════════════════════════════════════════════════════════
from preprocesamiento import FEATURES_EXTENDIDAS
cols_corr = [c for c in FEATURES_EXTENDIDAS if c in df.columns] + ["Churn"]
corr = df[cols_corr].corr()["Churn"].drop("Churn").sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colores = [VERDE if v < 0 else ROJO for v in corr.values]
ax.barh(corr.index, corr.values, color=colores, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlación de cada feature con Churn")
ax.set_xlabel("Correlación de Pearson")
for i, v in enumerate(corr.values):
    ax.text(v + 0.002 * np.sign(v), i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("exports/03_correlacion_churn.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico 3 guardado.")


# ═════════════════════════════════════════════════════════════
# GRÁFICO 4: Heatmap de correlaciones
# ═════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))
matriz = df[cols_corr].corr()
mask   = np.triu(np.ones_like(matriz, dtype=bool))
sns.heatmap(
    matriz, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, linewidths=0.5,
    ax=ax, annot_kws={"size": 8}
)
ax.set_title("Heatmap de correlaciones entre features")
plt.tight_layout()
plt.savefig("exports/04_heatmap_correlaciones.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico 4 guardado.")


# ═════════════════════════════════════════════════════════════
# GRÁFICO 5: Boxplots Churn vs No Churn
# ═════════════════════════════════════════════════════════════
features_box = ["PromedioAtraso", "TotalIncidencias",
                "TasaPuntualidad", "MesesComoCliente"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Comparación Churn vs No Churn — boxplots")

for i, feat in enumerate(features_box):
    if feat not in df.columns:
        continue
    data_plot = [
        df[df["Churn"] == 0][feat].dropna(),
        df[df["Churn"] == 1][feat].dropna()
    ]
    bp = axes[i].boxplot(data_plot, patch_artist=True,
                          labels=["No Churn", "Churn"],
                          medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor(VERDE)
    bp["boxes"][1].set_facecolor(ROJO)
    axes[i].set_title(feat)

plt.tight_layout()
plt.savefig("exports/05_boxplots_churn.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfico 5 guardado.")


# ═════════════════════════════════════════════════════════════
# GRÁFICO 6: Clientes por estado y zona
# ═════════════════════════════════════════════════════════════
if "Zona" in df.columns and "EstadoCliente" in df.columns:
    tabla = df.groupby(["Zona", "EstadoCliente"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    tabla.plot(kind="bar", ax=ax, edgecolor="white",
               color=[VERDE, ROJO, GRIS])
    ax.set_title("Distribución de clientes por zona y estado")
    ax.set_xlabel("Zona")
    ax.set_ylabel("Cantidad")
    ax.legend(title="Estado")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("exports/06_clientes_zona_estado.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Gráfico 6 guardado.")


print("\n" + "="*50)
print("  RESUMEN ESTADÍSTICO DEL DATASET")
print("="*50)
print(f"\n  Total clientes    : {len(df)}")
print(f"  Churn = 1         : {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
print(f"  Churn = 0         : {(df['Churn']==0).sum()}")
print(f"\n  Meses promedio    : {df['MesesComoCliente'].mean():.1f}")
print(f"  Atraso promedio   : {df['PromedioAtraso'].mean():.1f} días")
print(f"  Incidencias prom. : {df['TotalIncidencias'].mean():.1f}")
print(f"  Tasa puntualidad  : {df['TasaPuntualidad'].mean()*100:.1f}%")
print(f"\n  Exportados 6 gráficos en exports/")
