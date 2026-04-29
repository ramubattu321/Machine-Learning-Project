-- ============================================================
-- ML Production Optimization — SQL Analysis Queries (16)
-- Pre/post ML analysis on production_data.csv features
-- Run in: SQLite / MySQL / PostgreSQL / AWS Athena
-- Author: Ramu Battu — MS Data Analytics, CSU Fresno
-- ============================================================


-- ── QUERY 1: PRODUCTION SUMMARY BY MACHINE ───────────────────────────────────
-- Overall performance metrics per machine
SELECT
    machine_id,
    COUNT(*)                                            AS total_records,
    ROUND(AVG(temperature), 2)                          AS avg_temperature,
    ROUND(AVG(pressure), 2)                             AS avg_pressure,
    ROUND(AVG(humidity), 2)                             AS avg_humidity,
    ROUND(AVG(machine_hours), 2)                        AS avg_machine_hours,
    ROUND(AVG(defect_rate) * 100, 2)                    AS avg_defect_pct,
    ROUND(AVG(production_output), 0)                    AS avg_output,
    SUM(production_output)                              AS total_output
FROM production_data
GROUP BY machine_id
ORDER BY avg_defect_pct DESC;


-- ── QUERY 2: SHIFT PERFORMANCE COMPARISON ────────────────────────────────────
-- Which shift produces highest output with lowest defect rate?
SELECT
    shift,
    COUNT(*)                                            AS total_records,
    ROUND(AVG(temperature), 2)                          AS avg_temp,
    ROUND(AVG(humidity), 2)                             AS avg_humidity,
    ROUND(AVG(machine_hours), 2)                        AS avg_hours,
    ROUND(AVG(defect_rate) * 100, 2)                    AS avg_defect_pct,
    ROUND(AVG(production_output), 0)                    AS avg_output,
    SUM(production_output)                              AS total_output,
    RANK() OVER (ORDER BY AVG(production_output) DESC)  AS output_rank,
    RANK() OVER (ORDER BY AVG(defect_rate) ASC)         AS quality_rank
FROM production_data
GROUP BY shift
ORDER BY avg_output DESC;


-- ── QUERY 3: MODEL PERFORMANCE COMPARISON ────────────────────────────────────
-- Compare all ML models trained on production data
SELECT
    model_name,
    model_type,
    ROUND(mae, 2)                                       AS mae,
    ROUND(rmse, 2)                                      AS rmse,
    ROUND(r2_score, 4)                                  AS r2_score,
    ROUND(cv_r2, 4)                                     AS cv_r2_score,
    ROUND(training_time_s, 2)                           AS train_time_sec,
    RANK() OVER (ORDER BY r2_score DESC)                AS r2_rank,
    RANK() OVER (ORDER BY mae ASC)                      AS mae_rank,
    CASE WHEN r2_score >= 0.85 THEN 'Excellent'
         WHEN r2_score >= 0.75 THEN 'Good'
         WHEN r2_score >= 0.65 THEN 'Moderate'
         ELSE 'Needs Improvement' END                   AS performance_tier
FROM model_results
ORDER BY r2_rank;


-- ── QUERY 4: FEATURE IMPORTANCE COMPARISON ───────────────────────────────────
-- Which features drive production output most — Random Forest vs Gradient Boosting?
SELECT
    feature_name,
    MAX(CASE WHEN model_name='Random Forest'
             THEN importance_score END)                 AS rf_importance,
    MAX(CASE WHEN model_name='Gradient Boosting'
             THEN importance_score END)                 AS gb_importance,
    ROUND(AVG(importance_score), 4)                     AS avg_importance,
    MIN(rank_in_model)                                  AS best_rank
FROM feature_importance
GROUP BY feature_name
ORDER BY avg_importance DESC;


-- ── QUERY 5: TEMPERATURE IMPACT ON DEFECT RATE ───────────────────────────────
-- Bucket temperature and analyze defect rate per bucket
SELECT
    CASE
        WHEN temperature < 65 THEN 'Low (<65°C)'
        WHEN temperature < 75 THEN 'Normal (65-75°C)'
        WHEN temperature < 85 THEN 'High (75-85°C)'
        ELSE 'Very High (>85°C)'
    END                                                 AS temp_bucket,
    COUNT(*)                                            AS records,
    ROUND(AVG(defect_rate) * 100, 2)                    AS avg_defect_pct,
    ROUND(AVG(production_output), 0)                    AS avg_output,
    ROUND(MIN(defect_rate) * 100, 2)                    AS min_defect_pct,
    ROUND(MAX(defect_rate) * 100, 2)                    AS max_defect_pct
FROM production_data
GROUP BY temp_bucket
ORDER BY avg_defect_pct DESC;


-- ── QUERY 6: HUMIDITY IMPACT ON DEFECT RATE ──────────────────────────────────
-- Does high humidity increase defect rate?
SELECT
    CASE
        WHEN humidity < 45 THEN 'Low (<45%)'
        WHEN humidity < 55 THEN 'Normal (45-55%)'
        WHEN humidity < 65 THEN 'Elevated (55-65%)'
        ELSE 'High (>65%)'
    END                                                 AS humidity_bucket,
    COUNT(*)                                            AS records,
    ROUND(AVG(defect_rate) * 100, 2)                    AS avg_defect_pct,
    ROUND(AVG(production_output), 0)                    AS avg_output,
    ROUND(AVG(temperature), 2)                          AS avg_temp
FROM production_data
GROUP BY humidity_bucket
ORDER BY avg_defect_pct DESC;


-- ── QUERY 7: MACHINE HOURS vs OUTPUT CORRELATION ─────────────────────────────
-- How does machine hours impact production output?
SELECT
    CASE
        WHEN machine_hours < 6  THEN 'Short (<6 hrs)'
        WHEN machine_hours < 9  THEN 'Medium (6-9 hrs)'
        ELSE 'Long (>9 hrs)'
    END                                                 AS hours_bucket,
    COUNT(*)                                            AS records,
    ROUND(AVG(machine_hours), 2)                        AS avg_hours,
    ROUND(AVG(production_output), 0)                    AS avg_output,
    ROUND(AVG(defect_rate) * 100, 2)                    AS avg_defect_pct,
    SUM(production_output)                              AS total_output
FROM production_data
GROUP BY hours_bucket
ORDER BY avg_output DESC;


-- ── QUERY 8: DAILY OUTPUT TREND (WINDOW FUNCTION) ────────────────────────────
-- Daily total output with running cumulative
SELECT
    production_date,
    SUM(production_output)                              AS daily_output,
    SUM(defect_rate * production_output)                AS daily_defective_units,
    SUM(SUM(production_output)) OVER (
        ORDER BY production_date)                       AS cumulative_output,
    ROUND(AVG(defect_rate) * 100, 2)                    AS daily_avg_defect_pct
FROM production_data
GROUP BY production_date
ORDER BY production_date
LIMIT 15;


-- ── QUERY 9: MACHINE OUTPUT RANK PER SHIFT (RANK WINDOW) ────────────────────
-- Rank each machine by output within each shift
WITH machine_shift AS (
    SELECT
        machine_id,
        shift,
        ROUND(AVG(production_output), 0)  AS avg_output,
        ROUND(AVG(defect_rate) * 100, 2)  AS avg_defect_pct
    FROM production_data
    GROUP BY machine_id, shift
)
SELECT
    machine_id,
    shift,
    avg_output,
    avg_defect_pct,
    RANK() OVER (PARTITION BY shift ORDER BY avg_output DESC)    AS output_rank_in_shift,
    RANK() OVER (PARTITION BY shift ORDER BY avg_defect_pct ASC) AS quality_rank_in_shift
FROM machine_shift
ORDER BY shift, output_rank_in_shift;


-- ── QUERY 10: ROLLING 7-DAY AVERAGE OUTPUT ───────────────────────────────────
-- Smooth daily output using 7-day rolling window (LAG-based)
WITH daily AS (
    SELECT production_date,
           SUM(production_output)  AS daily_output,
           ROUND(AVG(defect_rate)*100, 2) AS daily_defect_pct
    FROM production_data
    GROUP BY production_date
)
SELECT
    production_date,
    daily_output,
    daily_defect_pct,
    LAG(daily_output, 1) OVER (ORDER BY production_date) AS prev_day_output,
    ROUND(daily_output - LAG(daily_output,1) OVER (ORDER BY production_date), 0) AS day_change,
    ROUND(100.0*(daily_output - LAG(daily_output,1) OVER (ORDER BY production_date))
          /NULLIF(LAG(daily_output,1) OVER (ORDER BY production_date),0), 1) AS dod_growth_pct
FROM daily
ORDER BY production_date
LIMIT 15;


-- ── QUERY 11: HIGH DEFECT RATE RECORDS — ANOMALY FLAGS ───────────────────────
-- Flag records with defect rate above 2x the mean (anomaly detection)
WITH stats AS (
    SELECT AVG(defect_rate) AS mean_defect,
           AVG(defect_rate) + 2*(AVG(defect_rate)) AS threshold
    FROM production_data
)
SELECT
    p.production_date,
    p.machine_id,
    p.shift,
    ROUND(p.temperature, 2)                             AS temperature,
    ROUND(p.humidity, 2)                                AS humidity,
    ROUND(p.defect_rate * 100, 2)                       AS defect_pct,
    ROUND(s.mean_defect * 100, 2)                       AS mean_defect_pct,
    'HIGH DEFECT ANOMALY'                               AS flag
FROM production_data p, stats s
WHERE p.defect_rate > s.threshold
ORDER BY p.defect_rate DESC
LIMIT 15;


-- ── QUERY 12: OPTIMAL OPERATING CONDITIONS ───────────────────────────────────
-- Find the parameter ranges that produce lowest defect rates
WITH ranked AS (
    SELECT *,
           NTILE(5) OVER (ORDER BY defect_rate ASC) AS defect_quintile
    FROM production_data
)
SELECT
    'Optimal (Bottom 20% Defect)' AS condition_label,
    ROUND(AVG(temperature), 2)    AS optimal_temp_range,
    ROUND(AVG(pressure), 2)       AS optimal_pressure,
    ROUND(AVG(humidity), 2)       AS optimal_humidity,
    ROUND(AVG(machine_hours), 2)  AS optimal_hours,
    ROUND(AVG(defect_rate)*100,2) AS avg_defect_pct,
    ROUND(AVG(production_output),0) AS avg_output,
    COUNT(*)                      AS record_count
FROM ranked
WHERE defect_quintile = 1
UNION ALL
SELECT
    'Poor (Top 20% Defect)',
    ROUND(AVG(temperature), 2), ROUND(AVG(pressure), 2),
    ROUND(AVG(humidity), 2), ROUND(AVG(machine_hours), 2),
    ROUND(AVG(defect_rate)*100,2), ROUND(AVG(production_output),0), COUNT(*)
FROM ranked WHERE defect_quintile = 5;


-- ── QUERY 13: CROSS-MACHINE DEFECT COMPARISON (NTILE) ────────────────────────
-- Classify each production record into defect severity quartiles
SELECT
    machine_id,
    shift,
    COUNT(*)                                                    AS total_records,
    SUM(CASE WHEN defect_quintile = 1 THEN 1 ELSE 0 END)       AS excellent_runs,
    SUM(CASE WHEN defect_quintile IN (2,3) THEN 1 ELSE 0 END)   AS normal_runs,
    SUM(CASE WHEN defect_quintile IN (4,5) THEN 1 ELSE 0 END)   AS poor_runs,
    ROUND(100.0*SUM(CASE WHEN defect_quintile IN(4,5) THEN 1 ELSE 0 END)/COUNT(*),1) AS poor_run_pct
FROM (
    SELECT *, NTILE(5) OVER (ORDER BY defect_rate ASC) AS defect_quintile
    FROM production_data
)
GROUP BY machine_id, shift
ORDER BY poor_run_pct DESC;


-- ── QUERY 14: MODEL R² IMPROVEMENT OVER BASELINE ─────────────────────────────
-- Show how much each model improves over linear regression baseline
WITH baseline AS (
    SELECT r2_score AS baseline_r2 FROM model_results
    WHERE model_name = 'Linear Regression'
)
SELECT
    m.model_name,
    m.model_type,
    ROUND(m.r2_score, 4)                                AS r2_score,
    ROUND(m.r2_score - b.baseline_r2, 4)               AS r2_improvement_over_baseline,
    ROUND(100.0*(m.r2_score-b.baseline_r2)/b.baseline_r2,1) AS pct_improvement,
    ROUND(m.mae, 2)                                     AS mae,
    ROUND(m.training_time_s, 2)                         AS train_time_sec,
    CASE WHEN m.r2_score = MAX(m.r2_score) OVER()
         THEN '✓ BEST MODEL' ELSE '' END                AS best_model_flag
FROM model_results m, baseline b
ORDER BY r2_score DESC;


-- ── QUERY 15: WEEKLY PRODUCTION EFFICIENCY ───────────────────────────────────
-- Aggregate production efficiency by week
SELECT
    STRFTIME('%Y-W%W', production_date)                AS week,
    COUNT(*)                                           AS records,
    SUM(production_output)                             AS weekly_output,
    ROUND(AVG(defect_rate)*100, 2)                     AS avg_defect_pct,
    ROUND(AVG(temperature), 2)                         AS avg_temp,
    ROUND(AVG(machine_hours), 2)                       AS avg_hours,
    ROUND(SUM(production_output)
          / NULLIF(SUM(machine_hours), 0), 1)          AS output_per_hour,
    RANK() OVER (ORDER BY SUM(production_output) DESC) AS output_rank
FROM production_data
GROUP BY week
ORDER BY week;


-- ── QUERY 16: PRODUCTION OPTIMIZATION DECISION SUPPORT ───────────────────────
-- Final CTE-based summary for model-driven production decisions
WITH production_stats AS (
    SELECT
        ROUND(AVG(production_output), 1)    AS mean_output,
        ROUND(AVG(defect_rate)*100, 2)      AS mean_defect_pct,
        ROUND(AVG(temperature), 2)          AS mean_temp,
        ROUND(AVG(machine_hours), 2)        AS mean_hours,
        SUM(production_output)              AS total_output
    FROM production_data
),
best_model AS (
    SELECT model_name, ROUND(r2_score,4) AS r2, ROUND(mae,2) AS mae
    FROM model_results ORDER BY r2_score DESC LIMIT 1
),
top_feature AS (
    SELECT feature_name, ROUND(AVG(importance_score),4) AS avg_importance
    FROM feature_importance GROUP BY feature_name ORDER BY avg_importance DESC LIMIT 1
),
best_machine AS (
    SELECT machine_id, ROUND(AVG(production_output),1) AS avg_out,
           ROUND(AVG(defect_rate)*100,2) AS defect_pct
    FROM production_data GROUP BY machine_id ORDER BY avg_out DESC LIMIT 1
),
best_shift AS (
    SELECT shift, ROUND(AVG(production_output),1) AS avg_out
    FROM production_data GROUP BY shift ORDER BY avg_out DESC LIMIT 1
)
SELECT
    ps.mean_output                          AS dataset_mean_output,
    ps.mean_defect_pct                      AS dataset_mean_defect_pct,
    ps.total_output                         AS total_units_produced,
    bm.model_name                           AS best_model,
    bm.r2                                   AS best_model_r2,
    bm.mae                                  AS best_model_mae,
    tf.feature_name                         AS top_driver_feature,
    tf.avg_importance                       AS feature_importance_score,
    bc.machine_id                           AS best_performing_machine,
    bc.avg_out                              AS machine_avg_output,
    bs.shift                                AS best_shift,
    bs.avg_out                              AS shift_avg_output
FROM production_stats ps, best_model bm, top_feature tf, best_machine bc, best_shift bs;
