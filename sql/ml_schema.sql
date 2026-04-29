-- ML Production Optimization — Database Schema
-- Features: Temperature, Pressure, Humidity, Machine Hours, Defect Rate, Production Output
-- Author: Ramu Battu — MS Data Analytics, CSU Fresno

DROP TABLE IF EXISTS feature_importance;
DROP TABLE IF EXISTS model_results;
DROP TABLE IF EXISTS production_data;

CREATE TABLE production_data (
    record_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    production_date  DATE    NOT NULL,
    machine_id       TEXT    NOT NULL,
    shift            TEXT    NOT NULL,
    temperature      REAL    NOT NULL,
    pressure         REAL    NOT NULL,
    humidity         REAL    NOT NULL,
    machine_hours    REAL    NOT NULL,
    defect_rate      REAL    NOT NULL,
    production_output INTEGER NOT NULL
);

CREATE TABLE model_results (
    model_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name       TEXT    NOT NULL,
    model_type       TEXT    NOT NULL,
    mae              REAL    NOT NULL,
    rmse             REAL    NOT NULL,
    r2_score         REAL    NOT NULL,
    cv_r2            REAL    NOT NULL,
    training_time_s  REAL    NOT NULL
);

CREATE TABLE feature_importance (
    feature_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name       TEXT    NOT NULL,
    feature_name     TEXT    NOT NULL,
    importance_score REAL    NOT NULL,
    rank_in_model    INTEGER NOT NULL
);

CREATE INDEX idx_prod_machine ON production_data(machine_id);
CREATE INDEX idx_prod_shift   ON production_data(shift);
CREATE INDEX idx_prod_date    ON production_data(production_date);
CREATE INDEX idx_model_name   ON model_results(model_name);
