-- ============================================================
-- ETF Strategy DB Health Check — CSV output
-- ============================================================

\copy (SELECT symbol, market, asset_type, is_core_training, is_cross_market, is_macro_proxy, history_policy_years, is_active FROM instrument_master ORDER BY asset_type, is_core_training DESC, symbol) TO 'query_result/check_instruments.csv' CSV HEADER

\copy (SELECT i.symbol, i.asset_type, MIN(b.trade_date) AS first_date, MAX(b.trade_date) AS last_date, COUNT(*) AS total_bars, MAX(b.trade_date) - MIN(b.trade_date) AS date_span_days FROM instrument_master i LEFT JOIN daily_bars b ON i.symbol = b.symbol GROUP BY i.symbol, i.asset_type ORDER BY i.asset_type, i.symbol) TO 'query_result/check_bar_coverage.csv' CSV HEADER

\copy (SELECT i.symbol, i.asset_type, i.is_active FROM instrument_master i LEFT JOIN daily_bars b ON i.symbol = b.symbol WHERE b.symbol IS NULL) TO 'query_result/check_no_bars.csv' CSV HEADER

\copy (SELECT i.asset_type, MAX(b.trade_date) AS latest_date, CURRENT_DATE - MAX(b.trade_date) AS days_stale FROM instrument_master i JOIN daily_bars b ON i.symbol = b.symbol GROUP BY i.asset_type ORDER BY i.asset_type) TO 'query_result/check_freshness.csv' CSV HEADER

\copy (SELECT symbol, fetch_start_date, fetch_end_date, rows_inserted, rows_updated, status, fetched_at FROM data_fetch_log ORDER BY fetched_at DESC LIMIT 20) TO 'query_result/check_fetch_log.csv' CSV HEADER

\copy (SELECT symbol, trade_date, COUNT(*) AS dupes FROM daily_bars GROUP BY symbol, trade_date HAVING COUNT(*) > 1 LIMIT 10) TO 'query_result/check_duplicates.csv' CSV HEADER

\copy (SELECT COUNT(*) AS total_bars, SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) AS null_open, SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_close, SUM(CASE WHEN adj_close IS NULL THEN 1 ELSE 0 END) AS null_adj_close, SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume FROM daily_bars b JOIN instrument_master i ON b.symbol = i.symbol WHERE i.asset_type = 'china_etf') TO 'query_result/check_nulls.csv' CSV HEADER

\echo 'Done — CSV files written to query_result/'
