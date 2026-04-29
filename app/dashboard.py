"""ETF 交互式仪表盘 — Streamlit 应用

启动:  streamlit run app/dashboard.py
需要:  ETF_DB_URL 环境变量指向 PostgreSQL

面板
----
1. 行情浏览   — 历史K线图          (筛选: ETF代码)
2. 价格预测   — 预测 vs 实际收盘价  (筛选: ETF代码 + 预测周期)
3. 市场总览   — 涨跌分布饼图
4. 预测精度   — 历史预测 vs 实际结果回测
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# 页面设置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ETF 仪表盘",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 中国市场配色 — 红涨绿跌
# ---------------------------------------------------------------------------
COLOR_UP = "#e74c3c"       # 红色 = 上涨
COLOR_DOWN = "#2ecc71"     # 绿色 = 下跌
COLOR_FLAT = "#95a5a6"     # 灰色 = 平盘
COLOR_UNKNOWN = "#bdc3c7"

COLOR_MAP = {
    "上涨": COLOR_UP,
    "下跌": COLOR_DOWN,
    "平盘": COLOR_FLAT,
    "未知": COLOR_UNKNOWN,
}

# ---------------------------------------------------------------------------
# 数据库连接
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 从项目根目录 .env 文件加载环境变量
load_dotenv(PROJECT_ROOT / ".env", override=True)


@st.cache_resource
def _engine():
    url = os.environ.get("ETF_DB_URL", "")
    if not url:
        try:
            from src.config import load_config

            cfg_path = PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml"
            cfg = load_config(cfg_path)
            if cfg.database.url_env:
                url = os.environ.get(cfg.database.url_env, "")
            if not url:
                url = cfg.database.url
        except Exception:
            pass
    if not url:
        st.error(
            "未找到数据库连接。请设置 `ETF_DB_URL` 环境变量，"
            "或在 TOML 配置文件中设置 `[database].url`。"
        )
        st.stop()
    return create_engine(url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# 缓存查询
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_symbols() -> pd.DataFrame:
    q = text("""
        SELECT symbol, name, asset_type, is_core_training
        FROM instrument_master
        WHERE is_active = true
        ORDER BY asset_type, symbol
    """)
    with _engine().connect() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(ttl=300)
def load_price_history(symbol: str) -> pd.DataFrame:
    q = text("""
        SELECT trade_date AS date, open, high, low, close, volume
        FROM daily_bars
        WHERE symbol = :sym
        ORDER BY trade_date
    """)
    with _engine().connect() as conn:
        return pd.read_sql(q, conn, params={"sym": symbol})


@st.cache_data(ttl=120)
def load_predictions(symbol: str) -> pd.DataFrame:
    q = text("""
        SELECT asof_date, current_close,
               y_close_1d_pred, y_close_3d_pred, y_close_5d_pred,
               model_version, data_freshness_date
        FROM prediction_snapshots
        WHERE symbol = :sym
        ORDER BY asof_date
    """)
    with _engine().connect() as conn:
        return pd.read_sql(q, conn, params={"sym": symbol})


@st.cache_data(ttl=120)
def load_latest_predictions() -> pd.DataFrame:
    q = text("""
        SELECT DISTINCT ON (symbol)
               symbol, asof_date, current_close,
               y_close_1d_pred, y_close_3d_pred, y_close_5d_pred,
               model_version, data_freshness_date
        FROM prediction_snapshots
        ORDER BY symbol, asof_date DESC
    """)
    with _engine().connect() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(ttl=600)
def load_model_freshness() -> dict:
    q = text("""
        SELECT model_version, train_start_date, train_end_date,
               val_start_date, val_end_date, created_at
        FROM model_runs
        WHERE model_name = 'dashboard'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "model_version": row.get("model_version", "unknown"),
        "train_start": str(row.get("train_start_date", "N/A")),
        "train_end": str(row.get("train_end_date", "N/A")),
        "val_end": str(row.get("val_end_date", "N/A")),
        "created_at": str(row.get("created_at", "N/A")),
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
FLAT_THRESHOLD = 0.001  # ±0.1% — ratio-based predictions are tightly clustered


def _classify_direction(current: float, predicted: float) -> str:
    if pd.isna(predicted) or pd.isna(current) or current == 0:
        return "未知"
    pct = (predicted - current) / current
    if pct > FLAT_THRESHOLD:
        return "上涨"
    elif pct < -FLAT_THRESHOLD:
        return "下跌"
    return "平盘"


def _direction_color_css(val: str) -> str:
    if val == "上涨":
        return f"color: {COLOR_UP}; font-weight: bold"
    elif val == "下跌":
        return f"color: {COLOR_DOWN}; font-weight: bold"
    return ""


HORIZON_DAYS = {"y_close_1d_pred": 1, "y_close_3d_pred": 3, "y_close_5d_pred": 5}


@st.cache_data(ttl=300)
def build_accuracy_dataset(symbol: str, horizon_col: str) -> pd.DataFrame:
    """Join past predictions with realized prices N trading days later.

    Returns DataFrame with columns:
        asof_date, predicted, actual, error, pct_error, pred_dir, actual_dir, dir_correct
    """
    preds = load_predictions(symbol)
    prices = load_price_history(symbol)
    if preds.empty or prices.empty:
        return pd.DataFrame()

    n_days = HORIZON_DAYS[horizon_col]
    prices["date"] = pd.to_datetime(prices["date"])
    preds["asof_date"] = pd.to_datetime(preds["asof_date"])

    # Build a map: for each trading date, what is the close price N trading days later?
    sorted_dates = prices.sort_values("date").reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates["date"])}

    rows = []
    for _, p in preds.iterrows():
        asof = p["asof_date"]
        predicted = p[horizon_col]
        current = p["current_close"]
        if pd.isna(predicted) or pd.isna(current) or asof not in date_to_idx:
            continue
        idx = date_to_idx[asof]
        target_idx = idx + n_days
        if target_idx >= len(sorted_dates):
            continue  # future date not yet available
        actual = sorted_dates.loc[target_idx, "close"]
        error = predicted - actual
        pct_error = error / actual * 100 if actual else 0
        # Direction: did price go up or down relative to current_close?
        pred_dir = "上涨" if predicted > current else ("下跌" if predicted < current else "平盘")
        actual_dir = "上涨" if actual > current else ("下跌" if actual < current else "平盘")
        rows.append({
            "asof_date": asof,
            "current_close": current,
            "predicted": predicted,
            "actual": actual,
            "error": error,
            "pct_error": pct_error,
            "pred_dir": pred_dir,
            "actual_dir": actual_dir,
            "dir_correct": pred_dir == actual_dir,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 构建 ETF 标签映射
# ---------------------------------------------------------------------------
symbols_df = load_symbols()
etf_symbols = symbols_df[symbols_df["asset_type"] == "china_etf"]["symbol"].tolist()

if not etf_symbols:
    st.warning("数据库中未找到中国ETF品种。")
    st.stop()

sym_labels: dict[str, str] = {}
for _, row in symbols_df[symbols_df["asset_type"] == "china_etf"].iterrows():
    label = row["symbol"]
    if pd.notna(row.get("name")) and row["name"]:
        label = f"{row['symbol']} — {row['name']}"
    sym_labels[label] = row["symbol"]

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.title("ETF 仪表盘")
st.sidebar.caption("行情浏览 · 价格预测 · 市场总览")

# ---------------------------------------------------------------------------
# 标签页
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["行情浏览", "价格预测", "市场总览", "预测精度"])

# ========================= 标签1: 行情浏览 ================================
with tab1:
    st.header("行情浏览")

    col_filter, col_chart = st.columns([1, 3])

    with col_filter:
        selected_label = st.selectbox(
            "选择ETF",
            options=list(sym_labels.keys()),
            key="price_etf",
        )
        selected_sym = sym_labels[selected_label]

    prices = load_price_history(selected_sym)

    with col_chart:
        if prices.empty:
            st.info(f"{selected_sym} 暂无行情数据。")
        else:
            # K线图 — 红涨绿跌
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=prices["date"],
                open=prices["open"],
                high=prices["high"],
                low=prices["low"],
                close=prices["close"],
                name=selected_sym,
                increasing_line_color=COLOR_UP,
                increasing_fillcolor=COLOR_UP,
                decreasing_line_color=COLOR_DOWN,
                decreasing_fillcolor=COLOR_DOWN,
            ))
            fig.update_layout(
                title=f"{selected_sym} — 历史K线",
                xaxis_title="日期",
                yaxis_title="价格 (元)",
                xaxis_rangeslider_visible=False,
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 统计指标 — 2x2 布局
            latest = prices.iloc[-1]
            row1_c1, row1_c2 = st.columns(2)
            row2_c1, row2_c2 = st.columns(2)

            row1_c1.metric("最新收盘价", f"¥{latest['close']:.3f}")
            if len(prices) > 1:
                ret = (prices.iloc[-1]["close"] / prices.iloc[0]["close"] - 1) * 100
                ret_color = COLOR_UP if ret > 0 else COLOR_DOWN
                row1_c2.markdown(
                    f"**区间总回报**<br>"
                    f"<span style='font-size:1.8rem; color:{ret_color}'>{ret:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
            else:
                row1_c2.metric("区间总回报", "N/A")

            date_min = str(prices["date"].min())[:10]
            date_max = str(prices["date"].max())[:10]
            row2_c1.metric("数据区间", f"{date_min} ~ {date_max}")
            row2_c2.metric("交易日数", f"{len(prices):,}")

# ========================= 标签2: 价格预测 ================================
with tab2:
    st.header("收盘价预测")

    col_f1, col_f2, _ = st.columns([1, 1, 2])

    with col_f1:
        pred_label = st.selectbox(
            "选择ETF",
            options=list(sym_labels.keys()),
            key="pred_etf",
        )
        pred_sym = sym_labels[pred_label]

    with col_f2:
        horizon_map = {
            "1日预测": "y_close_1d_pred",
            "3日预测": "y_close_3d_pred",
            "5日预测": "y_close_5d_pred",
        }
        horizon_label = st.selectbox(
            "预测周期",
            options=list(horizon_map.keys()),
            key="pred_horizon",
        )
        horizon_col = horizon_map[horizon_label]
        horizon_n = HORIZON_DAYS[horizon_col]

    preds = load_predictions(pred_sym)

    if preds.empty:
        st.info(
            f"{pred_sym} 暂无预测数据。"
        )
    else:
        chart_df = preds[["asof_date", "current_close", horizon_col]].dropna().copy()

        if chart_df.empty:
            st.info(f"{pred_sym} 暂无{horizon_label}数据。")
        else:
            # --- 获取最新预测行 ---
            latest_row = chart_df.iloc[-1]
            asof = pd.Timestamp(latest_row["asof_date"])
            pred_price = latest_row[horizon_col]
            cur_price = latest_row["current_close"]

            # --- 构建预测目标日期（向前推 N 个交易日，跳过周末）---
            future_date = asof
            days_added = 0
            while days_added < horizon_n:
                future_date += pd.Timedelta(days=1)
                if future_date.weekday() < 5:  # 周一~周五
                    days_added += 1

            # --- 获取 asof_date 前 5 个交易日的历史收盘价 ---
            hist_prices = load_price_history(pred_sym)
            if not hist_prices.empty:
                hist_prices["date"] = pd.to_datetime(hist_prices["date"])
                hist_tail = hist_prices[hist_prices["date"] <= asof].tail(5)
            else:
                hist_tail = pd.DataFrame()

            # --- 构建图表 ---
            fig = go.Figure()

            # 历史收盘价（最近5个交易日）
            if not hist_tail.empty:
                fig.add_trace(go.Scatter(
                    x=hist_tail["date"],
                    y=hist_tail["close"],
                    mode="lines+markers",
                    name="近期收盘价",
                    line=dict(color="#636EFA", width=2),
                    marker=dict(size=5),
                ))

            # 从 asof_date 的 current_close 到 future_date 的 predicted close 画连线
            fig.add_trace(go.Scatter(
                x=[asof, future_date],
                y=[cur_price, pred_price],
                mode="lines+markers",
                name=f"预测 ({horizon_label})",
                line=dict(
                    color=COLOR_UP if pred_price >= cur_price else COLOR_DOWN,
                    width=2.5,
                    dash="dot",
                ),
                marker=dict(size=8, symbol="diamond"),
            ))

            # 标注预测点
            fig.add_annotation(
                x=future_date,
                y=pred_price,
                text=f"¥{pred_price:.3f}",
                showarrow=True,
                arrowhead=2,
                ax=-40,
                ay=-30,
                font=dict(size=12),
            )

            fig.update_layout(
                title=f"{pred_sym} — {horizon_label}（预测日: {str(future_date.date())}）",
                xaxis_title="日期",
                yaxis_title="价格 (元)",
                height=480,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 最新预测摘要
            direction = _classify_direction(cur_price, pred_price)
            pct_chg = ((pred_price - cur_price) / cur_price * 100) if cur_price else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("当前收盘价", f"¥{cur_price:.3f}")
            m2.metric(f"预测价 ({horizon_label})", f"¥{pred_price:.3f}")
            delta_color = "normal" if pct_chg < 0 else "inverse"
            m3.metric(
                "预期变动",
                f"{pct_chg:+.2f}%",
                delta=direction,
                delta_color=delta_color,
            )

    # --- 模型时效声明 ---
    st.divider()
    freshness = load_model_freshness()
    if freshness:
        st.caption(
            f"模型 **{freshness.get('model_version', 'N/A')}** — "
            f"训练数据区间: **{freshness.get('train_start', 'N/A')}** "
            f"至 **{freshness.get('train_end', 'N/A')}**"
            f"（验证集截至 {freshness.get('val_end', 'N/A')}）。"
            f"训练时间: {freshness.get('created_at', 'N/A')}。"
        )
    else:
        if not preds.empty and "data_freshness_date" in preds.columns:
            fresh_date = preds["data_freshness_date"].dropna()
            mv = preds["model_version"].dropna()
            if not fresh_date.empty:
                st.caption(
                    f"模型 **{mv.iloc[-1] if not mv.empty else 'N/A'}** — "
                    f"预测基于截至 **{fresh_date.iloc[-1]}** 的市场数据。"
                )
        else:
            st.caption("模型时效信息暂不可用。")

# ========================= 标签3: 市场总览 ================================
with tab3:
    st.header("市场总览")
    st.caption("基于最新模型预测的ETF涨跌分布。")

    col_h, _ = st.columns([1, 3])
    with col_h:
        overview_horizon_map = {
            "1日预测": "y_close_1d_pred",
            "3日预测": "y_close_3d_pred",
            "5日预测": "y_close_5d_pred",
        }
        overview_horizon_label = st.selectbox(
            "预测周期",
            options=list(overview_horizon_map.keys()),
            key="overview_horizon",
        )
        overview_col = overview_horizon_map[overview_horizon_label]

    latest_preds = load_latest_predictions()

    if latest_preds.empty:
        st.info(
            "数据库中暂无预测数据。"
        )
    else:
        # 分类涨跌 + 计算价差
        latest_preds["方向"] = latest_preds.apply(
            lambda r: _classify_direction(r["current_close"], r[overview_col]),
            axis=1,
        )
        latest_preds["价差"] = latest_preds[overview_col] - latest_preds["current_close"]

        direction_counts = latest_preds["方向"].value_counts().reset_index()
        direction_counts.columns = ["方向", "数量"]

        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            fig = px.pie(
                direction_counts,
                values="数量",
                names="方向",
                color="方向",
                color_discrete_map=COLOR_MAP,
                title=f"ETF涨跌分布（{overview_horizon_label}）",
                hole=0.35,
            )
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("明细")
            detail_df = latest_preds[
                ["symbol", "asof_date", "current_close", overview_col, "价差", "方向"]
            ].copy()
            detail_df.columns = [
                "代码", "预测日期", "当前收盘价",
                f"预测价（{overview_horizon_label}）", "价差（预测-当前）", "方向",
            ]
            detail_df = detail_df.sort_values("价差（预测-当前）", ascending=False)

            st.dataframe(
                detail_df.style
                .map(_direction_color_css, subset=["方向"])
                .format({"当前收盘价": "¥{:.3f}",
                         f"预测价（{overview_horizon_label}）": "¥{:.3f}",
                         "价差（预测-当前）": "{:+.3f}"}),
                use_container_width=True,
                height=400,
            )

        # 汇总指标
        n_total = len(latest_preds)
        n_up = len(latest_preds[latest_preds["方向"] == "上涨"])
        n_down = len(latest_preds[latest_preds["方向"] == "下跌"])
        n_flat = len(latest_preds[latest_preds["方向"] == "平盘"])

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("ETF总数", n_total)
        s2.metric("上涨", n_up)
        s3.metric("下跌", n_down)
        s4.metric("平盘", n_flat)

# ========================= 标签4: 预测精度 ================================
with tab4:
    st.header("预测精度回测")
    st.caption("将历史预测与实际成交价对比，衡量模型表现。")

    col_a1, col_a2, _ = st.columns([1, 1, 2])

    with col_a1:
        acc_label = st.selectbox(
            "选择ETF",
            options=list(sym_labels.keys()),
            key="acc_etf",
        )
        acc_sym = sym_labels[acc_label]

    with col_a2:
        acc_horizon_map = {
            "1日预测": "y_close_1d_pred",
            "3日预测": "y_close_3d_pred",
            "5日预测": "y_close_5d_pred",
        }
        acc_horizon_label = st.selectbox(
            "预测周期",
            options=list(acc_horizon_map.keys()),
            key="acc_horizon",
        )
        acc_horizon_col = acc_horizon_map[acc_horizon_label]

    acc_df = build_accuracy_dataset(acc_sym, acc_horizon_col)

    if acc_df.empty:
        st.info(
            f"{acc_sym} 暂无可回测的预测数据。"
            "需要已有预测记录且对应的实际行情数据已入库。"
        )
    else:
        # --- 汇总指标 ---
        mae = acc_df["error"].abs().mean()
        mape = acc_df["pct_error"].abs().mean()
        dir_acc = acc_df["dir_correct"].mean() * 100
        n_samples = len(acc_df)
        corr = acc_df["predicted"].corr(acc_df["actual"])

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("样本数", n_samples)
        k2.metric("平均绝对误差", f"¥{mae:.3f}")
        k3.metric("平均百分比误差", f"{mape:.2f}%")
        k4.metric("方向准确率", f"{dir_acc:.1f}%")
        k5.metric("预测相关系数", f"{corr:.3f}")

        # --- 图表行 ---
        chart_col1, chart_col2 = st.columns(2)

        # 散点图: 预测 vs 实际
        with chart_col1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=acc_df["actual"],
                y=acc_df["predicted"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=acc_df["dir_correct"].map({True: "#636EFA", False: "#EF553B"}),
                    opacity=0.7,
                ),
                text=acc_df["asof_date"].dt.strftime("%Y-%m-%d"),
                hovertemplate="实际: ¥%{x:.3f}<br>预测: ¥%{y:.3f}<br>日期: %{text}",
                name="预测点",
            ))
            # 对角线 (完美预测)
            axis_min = min(acc_df["actual"].min(), acc_df["predicted"].min())
            axis_max = max(acc_df["actual"].max(), acc_df["predicted"].max())
            fig_scatter.add_trace(go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                name="完美预测线",
                showlegend=True,
            ))
            fig_scatter.update_layout(
                title="预测价 vs 实际价",
                xaxis_title="实际收盘价 (元)",
                yaxis_title="预测收盘价 (元)",
                height=400,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # 时序图: 误差随时间变化
        with chart_col2:
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=acc_df["asof_date"],
                y=acc_df["error"],
                marker_color=acc_df["error"].apply(
                    lambda e: COLOR_UP if e > 0 else COLOR_DOWN
                ),
                name="预测误差",
                hovertemplate="日期: %{x}<br>误差: ¥%{y:.3f}",
            ))
            fig_err.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig_err.update_layout(
                title="预测误差时序（预测 - 实际）",
                xaxis_title="预测日期",
                yaxis_title="误差 (元)",
                height=400,
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # --- 明细表 ---
        st.subheader("历史预测明细")
        detail = acc_df[
            ["asof_date", "current_close", "predicted", "actual",
             "error", "pct_error", "pred_dir", "actual_dir", "dir_correct"]
        ].copy()
        detail.columns = [
            "预测日期", "当时收盘价", "预测价", "实际价",
            "误差", "误差%", "预测方向", "实际方向", "方向正确",
        ]
        detail = detail.sort_values("预测日期", ascending=False)

        st.dataframe(
            detail.style
            .format({
                "当时收盘价": "¥{:.3f}",
                "预测价": "¥{:.3f}",
                "实际价": "¥{:.3f}",
                "误差": "{:+.3f}",
                "误差%": "{:+.2f}%",
            })
            .map(
                lambda v: f"color: {COLOR_UP}; font-weight: bold" if v == "上涨"
                else (f"color: {COLOR_DOWN}; font-weight: bold" if v == "下跌" else ""),
                subset=["预测方向", "实际方向"],
            )
            .map(
                lambda v: "background-color: rgba(46,204,113,0.15)" if v is True
                else ("background-color: rgba(231,76,60,0.15)" if v is False else ""),
                subset=["方向正确"],
            ),
            use_container_width=True,
            height=400,
        )
