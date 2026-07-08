-- ============================================================
-- Cable Visión IA — Script de creación de base de datos
-- UPSJB — Inteligencia Artificial 2026
-- Ejecutar en SQL Server Management Studio
-- ============================================================

USE master;
GO

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'CableVisionIA')
    CREATE DATABASE CableVisionIA;
GO

USE CableVisionIA;
GO

-- ============================================================
-- TABLAS BASE
-- ============================================================

-- Clientes
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Clientes' AND xtype='U')
CREATE TABLE Clientes (
    ClienteID           INT IDENTITY(1,1) PRIMARY KEY,
    Nombre              NVARCHAR(100)   NOT NULL,
    Zona                NVARCHAR(50),
    PlanContratado      NVARCHAR(50),
    PrecioMensual       DECIMAL(10,2),
    FechaAlta           DATE,
    EstadoCliente       NVARCHAR(20)    DEFAULT 'Activo',  -- Activo / Inactivo
    Churn               BIT             DEFAULT 0
);
GO

-- Pagos
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Pagos' AND xtype='U')
CREATE TABLE Pagos (
    PagoID              INT IDENTITY(1,1) PRIMARY KEY,
    ClienteID           INT             NOT NULL REFERENCES Clientes(ClienteID),
    FechaVencimiento    DATE            NOT NULL,
    FechaPago           DATE,
    Monto               DECIMAL(10,2),
    DiasAtraso          INT             DEFAULT 0,
    Estado              NVARCHAR(20)    DEFAULT 'Pendiente'  -- Puntual / Vencido / Pendiente
);
GO

-- Incidencias técnicas
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Incidencias' AND xtype='U')
CREATE TABLE Incidencias (
    IncidenciaID        INT IDENTITY(1,1) PRIMARY KEY,
    ClienteID           INT             NOT NULL REFERENCES Clientes(ClienteID),
    Zona                NVARCHAR(50),
    FechaReporte        DATETIME        DEFAULT GETDATE(),
    FechaResolucion     DATETIME,
    TiempoResolucionMin INT,
    Calificacion        DECIMAL(3,2)    CHECK (Calificacion BETWEEN 1 AND 5)
);
GO

-- Atenciones al cliente (soporte)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Atenciones' AND xtype='U')
CREATE TABLE Atenciones (
    AtencionID          INT IDENTITY(1,1) PRIMARY KEY,
    ClienteID           INT             NOT NULL REFERENCES Clientes(ClienteID),
    FechaAtencion       DATETIME        DEFAULT GETDATE(),
    Canal               NVARCHAR(30)
);
GO

-- Equipos / Inventario
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Equipos' AND xtype='U')
CREATE TABLE Equipos (
    EquipoID            INT IDENTITY(1,1) PRIMARY KEY,
    NombreEquipo        NVARCHAR(100)   NOT NULL,
    Categoria           NVARCHAR(50),
    StockActual         INT             DEFAULT 0,
    StockMinimo         INT             DEFAULT 5
);
GO

-- Instalaciones por zona (demanda histórica)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Instalaciones' AND xtype='U')
CREATE TABLE Instalaciones (
    InstalacionID       INT IDENTITY(1,1) PRIMARY KEY,
    Zona                NVARCHAR(50)    NOT NULL,
    Periodo             DATE            NOT NULL,  -- primer día del mes
    CantidadSolicitudes INT             DEFAULT 0,
    CantidadRealizadas  INT             DEFAULT 0
);
GO

-- Modelos IA registrados
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ModelosIA' AND xtype='U')
CREATE TABLE ModelosIA (
    ModeloID    INT IDENTITY(1,1) PRIMARY KEY,
    Nombre      NVARCHAR(100)   NOT NULL,
    Version     NVARCHAR(20),
    Algoritmo   NVARCHAR(50),
    CasoDeUso   NVARCHAR(100),
    Precision   DECIMAL(6,4),
    Recall      DECIMAL(6,4),
    F1Score     DECIMAL(6,4),
    AUC_ROC     DECIMAL(6,4),
    RutaModelo  NVARCHAR(255),
    FechaEntrenamiento DATETIME DEFAULT GETDATE(),
    Activo      BIT DEFAULT 1
);
GO

-- Predicciones de churn
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='PrediccionesChurn' AND xtype='U')
CREATE TABLE PrediccionesChurn (
    PrediccionID    INT IDENTITY(1,1) PRIMARY KEY,
    ModeloID        INT             NOT NULL REFERENCES ModelosIA(ModeloID),
    ClienteID       INT             NOT NULL REFERENCES Clientes(ClienteID),
    FechaPrediccion DATETIME        DEFAULT GETDATE(),
    ScoreChurn      DECIMAL(6,4),
    NivelRiesgo     NVARCHAR(20)    -- Alto / Medio / Bajo
);
GO

-- Segmentos K-Means
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='SegmentosCliente' AND xtype='U')
CREATE TABLE SegmentosCliente (
    SegmentoID      INT IDENTITY(1,1) PRIMARY KEY,
    ModeloID        INT             NOT NULL REFERENCES ModelosIA(ModeloID),
    ClienteID       INT             NOT NULL REFERENCES Clientes(ClienteID),
    FechaAsignacion DATETIME        DEFAULT GETDATE(),
    ClusterID       INT,
    NombreSegmento  NVARCHAR(50)
);
GO

-- ============================================================
-- VISTAS
-- ============================================================

-- vw_PerfilClienteChurn: features para modelo de churn
CREATE OR ALTER VIEW vw_PerfilClienteChurn AS
SELECT
    c.ClienteID,
    c.EstadoCliente,
    c.Churn,
    DATEDIFF(MONTH, c.FechaAlta, GETDATE())                         AS MesesComoCliente,
    ISNULL(p.TotalPagos,       0)                                   AS TotalPagos,
    ISNULL(p.PagosPuntuales,   0)                                   AS PagosPuntuales,
    ISNULL(p.PagosVencidos,    0)                                   AS PagosVencidos,
    ISNULL(p.PromedioAtraso,   0)                                   AS PromedioAtraso,
    ISNULL(i.TotalIncidencias, 0)                                   AS TotalIncidencias,
    ISNULL(i.PromedioResMin,   0)                                   AS PromedioResolucionMin,
    ISNULL(i.PromedioCalif,    0)                                   AS PromedioCalificacion,
    ISNULL(a.TotalAtenciones,  0)                                   AS TotalAtenciones,
    ISNULL(c.PrecioMensual,    0)                                   AS PrecioMensual,
    -- Features derivadas
    CASE WHEN ISNULL(p.TotalPagos,0) > 0
         THEN CAST(ISNULL(p.PagosPuntuales,0) AS FLOAT) / p.TotalPagos
         ELSE 0 END                                                 AS TasaPuntualidad,
    CASE WHEN DATEDIFF(MONTH, c.FechaAlta, GETDATE()) > 0
         THEN CAST(ISNULL(i.TotalIncidencias,0) AS FLOAT) / DATEDIFF(MONTH, c.FechaAlta, GETDATE())
         ELSE 0 END                                                 AS IncidenciasPorMes,
    ISNULL(i.PromedioCalif, 0) *
        CASE WHEN ISNULL(p.TotalPagos,0) > 0
             THEN CAST(ISNULL(p.PagosPuntuales,0) AS FLOAT) / p.TotalPagos
             ELSE 0 END                                             AS IndSatisfaccion,
    CASE WHEN ISNULL(p.PagosVencidos,0) > 0 THEN 1 ELSE 0 END      AS TieneDeuda,
    CASE WHEN ISNULL(i.TotalIncidencias,0) >= 3 THEN 1 ELSE 0 END  AS MultiplesIncidencias,
    ISNULL(p.PromedioAtraso,0) * ISNULL(p.PagosVencidos,0)         AS AtrasoPonderado
FROM Clientes c
LEFT JOIN (
    SELECT ClienteID,
           COUNT(*)                             AS TotalPagos,
           SUM(CASE WHEN Estado='Puntual' THEN 1 ELSE 0 END) AS PagosPuntuales,
           SUM(CASE WHEN Estado='Vencido' THEN 1 ELSE 0 END) AS PagosVencidos,
           AVG(CAST(DiasAtraso AS FLOAT))       AS PromedioAtraso
    FROM Pagos
    GROUP BY ClienteID
) p ON c.ClienteID = p.ClienteID
LEFT JOIN (
    SELECT ClienteID,
           COUNT(*)                             AS TotalIncidencias,
           AVG(CAST(TiempoResolucionMin AS FLOAT)) AS PromedioResMin,
           AVG(Calificacion)                    AS PromedioCalif
    FROM Incidencias
    GROUP BY ClienteID
) i ON c.ClienteID = i.ClienteID
LEFT JOIN (
    SELECT ClienteID,
           COUNT(*) AS TotalAtenciones
    FROM Atenciones
    GROUP BY ClienteID
) a ON c.ClienteID = a.ClienteID;
GO

-- vw_DemandaInstalacionesMensual
CREATE OR ALTER VIEW vw_DemandaInstalacionesMensual AS
SELECT
    Zona,
    Periodo,
    CantidadSolicitudes,
    CantidadRealizadas,
    CASE WHEN CantidadSolicitudes > 0
         THEN CAST(CantidadRealizadas AS FLOAT) / CantidadSolicitudes
         ELSE 0 END AS TasaCumplimiento
FROM Instalaciones;
GO

-- vw_IncidenciasPorZona
CREATE OR ALTER VIEW vw_IncidenciasPorZona AS
SELECT
    Zona,
    COUNT(*)                              AS TotalIncidencias,
    AVG(CAST(TiempoResolucionMin AS FLOAT)) AS PromedioResolucionMin,
    AVG(Calificacion)                     AS PromedioCalificacion,
    MONTH(FechaReporte)                   AS Mes,
    YEAR(FechaReporte)                    AS Anio
FROM Incidencias
GROUP BY Zona, MONTH(FechaReporte), YEAR(FechaReporte);
GO

-- vw_AlertaStockCritico
CREATE OR ALTER VIEW vw_AlertaStockCritico AS
SELECT
    EquipoID,
    NombreEquipo,
    Categoria,
    StockActual,
    StockMinimo,
    StockActual - StockMinimo                   AS DiferenciaMinimo,
    CASE
        WHEN StockActual = 0              THEN 'CRITICO'
        WHEN StockActual < StockMinimo    THEN 'BAJO'
        ELSE 'OK'
    END AS EstadoStock
FROM Equipos;
GO

-- ============================================================
-- DATOS DE EJEMPLO (opcional — borrar en producción)
-- ============================================================

-- Modelos base
INSERT INTO ModelosIA (Nombre, Version, Algoritmo, CasoDeUso, Precision, Recall, F1Score, AUC_ROC, RutaModelo)
VALUES
    ('Churn XGBoost',       'v1', 'XGBoost',          'Predicción de churn',     0, 0, 0, 0, 'models/churn_xgb_v1.pkl'),
    ('Churn Random Forest', 'v2', 'Random Forest',     'Churn (mejor recall)',    0, 0, 0, 0, 'models/churn_rf_v2.pkl'),
    ('Fallas Isolation',    'v1', 'Isolation Forest',  'Detección de anomalías',  0, 0, 0, 0, 'models/fallas_if_v1.pkl'),
    ('Segmentación KMeans', 'v1', 'K-Means',           'Segmentación clientes',   0, 0, 0, 0, 'models/segmento_km_v1.pkl'),
    ('KMeans Segmentos',    'v1', 'K-Means',           'Segmentación (alt)',      0, 0, 0, 0, 'models/segmento_km_v1.pkl');
GO

PRINT 'Base de datos CableVisionIA creada correctamente.';
GO
