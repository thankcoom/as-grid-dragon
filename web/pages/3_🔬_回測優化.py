"""
回測/優化頁面
=============
回測、參數優化、結果分析
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="回測優化 - AS 網格",
    page_icon="🔬",
    layout="wide",
)

# 導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from theme import apply_custom_theme
from components.sidebar import render_sidebar
apply_custom_theme()

from state import init_session_state, get_config, save_config
from config.models import SymbolConfig
from utils import normalize_symbol
from core.backtest import BacktestManager

init_session_state()


@st.cache_resource
def get_backtest_manager():
    """取得回測管理器 (快取)"""
    return BacktestManager()


def render_symbol_input():
    """渲染交易對輸入"""
    st.subheader("📊 選擇交易對")

    config = get_config()

    # 選擇已有或輸入新的
    tab1, tab2 = st.tabs(["已配置交易對", "自訂交易對"])

    with tab1:
        if config.symbols:
            symbol = st.selectbox(
                "選擇交易對",
                options=list(config.symbols.keys()),
                key="existing_symbol"
            )
            if symbol:
                cfg = config.symbols[symbol]
                st.caption(f"止盈: {cfg.take_profit_spacing*100:.2f}% | 補倉: {cfg.grid_spacing*100:.2f}%")
                return symbol, cfg.ccxt_symbol, cfg
        else:
            st.info("尚未配置交易對")

    with tab2:
        symbol_input = st.text_input("輸入交易對", placeholder="例如: XRPUSDC")
        if symbol_input:
            raw, ccxt_sym, coin, quote = normalize_symbol(symbol_input)
            if raw:
                st.success(f"識別為: {coin}/{quote}")
                # 使用預設配置
                cfg = SymbolConfig(symbol=raw, ccxt_symbol=ccxt_sym)
                return raw, ccxt_sym, cfg
            else:
                st.error("無法識別交易對格式")

    return None, None, None


def render_date_range():
    """渲染日期範圍選擇"""
    st.subheader("📅 選擇日期範圍")

    col1, col2 = st.columns(2)

    today = datetime.now()

    with col1:
        preset = st.radio(
            "快速選擇",
            options=["最近 7 天", "最近 30 天", "最近 90 天", "自訂"],
            horizontal=True,
        )

    days_map = {"最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}

    if preset in days_map:
        days = days_map[preset]
        end_date = (today - timedelta(days=1)).date()
        start_date = (today - timedelta(days=days)).date()
    else:
        with col2:
            start_date = st.date_input(
                "開始日期",
                value=(today - timedelta(days=30)).date()
            )
            end_date = st.date_input(
                "結束日期",
                value=(today - timedelta(days=1)).date()
            )

    return str(start_date), str(end_date)


def render_backtest_params(sym_config: SymbolConfig):
    """渲染回測參數"""
    st.subheader("⚙️ 回測參數")

    col1, col2 = st.columns(2)

    with col1:
        take_profit = st.number_input(
            "止盈間距 (%)",
            min_value=0.1,
            max_value=5.0,
            value=sym_config.take_profit_spacing * 100,
            step=0.1,
        )

        grid_spacing = st.number_input(
            "補倉間距 (%)",
            min_value=0.1,
            max_value=5.0,
            value=sym_config.grid_spacing * 100,
            step=0.1,
        )

    with col2:
        quantity = st.number_input(
            "每單數量",
            min_value=1.0,
            value=float(sym_config.initial_quantity),
            step=1.0,
        )

        leverage = st.number_input(
            "槓桿",
            min_value=1,
            max_value=125,
            value=sym_config.leverage,
        )

    # 更新配置
    sym_config.take_profit_spacing = take_profit / 100
    sym_config.grid_spacing = grid_spacing / 100
    sym_config.initial_quantity = quantity
    sym_config.leverage = leverage

    return sym_config


def run_single_backtest(manager: BacktestManager, symbol: str, ccxt_symbol: str,
                        sym_config: SymbolConfig, start_date: str, end_date: str):
    """執行單筆回測"""
    # 檢查並下載數據
    available_dates = manager.get_available_dates(symbol)

    with st.spinner("檢查數據..."):
        # 計算需要的日期
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days + 1

        need_download = any(
            (start + timedelta(days=i)).strftime("%Y-%m-%d") not in available_dates
            for i in range(days)
        )

        if need_download:
            st.info("下載歷史數據中...")
            manager.download_data(symbol, ccxt_symbol, start_date, end_date)

    # 載入數據
    with st.spinner("載入數據..."):
        df = manager.load_data(symbol, start_date, end_date)

    if df is None or df.empty:
        st.error("載入數據失敗")
        return None

    st.success(f"載入 {len(df):,} 條 K 線")

    # 執行回測
    with st.spinner("執行回測..."):
        result = manager.run_backtest(sym_config, df)

    return result


def render_backtest_result(result: dict):
    """渲染回測結果"""
    st.subheader("📈 回測結果")

    # 收益概況
    col1, col2, col3, col4 = st.columns(4)

    ret_pct = result.get('return_pct', 0) * 100
    color = "normal" if ret_pct >= 0 else "inverse"

    with col1:
        st.metric(
            "收益率",
            f"{ret_pct:+.2f}%",
            delta=f"{ret_pct:+.2f}%" if ret_pct != 0 else None,
            delta_color=color
        )

    with col2:
        st.metric(
            "最終淨值",
            f"{result.get('final_equity', 0):.2f} U"
        )

    with col3:
        st.metric(
            "最大回撤",
            f"{result.get('max_drawdown', 0)*100:.2f}%"
        )

    with col4:
        st.metric(
            "勝率",
            f"{result.get('win_rate', 0)*100:.1f}%"
        )

    # 交易統計
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("總交易數", result.get('trades_count', 0))

    with col2:
        st.metric("多單成交", result.get('long_trades', 0))

    with col3:
        st.metric("空單成交", result.get('short_trades', 0))

    with col4:
        pf = result.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric("盈虧比", pf_str)

    # 收益曲線
    equity_curve = result.get('equity_curve', [])
    if equity_curve:
        st.divider()
        st.markdown("**收益曲線**")

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='淨值',
            line=dict(color='#00CC96', width=2)
        ))
        fig.add_hline(y=1000, line_dash="dash", line_color="gray",
                      annotation_text="初始資金")
        fig.update_layout(
            xaxis_title="K 線數",
            yaxis_title="淨值 (U)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    return result


def run_optimization(manager: BacktestManager, symbol: str, ccxt_symbol: str,
                     sym_config: SymbolConfig, start_date: str, end_date: str):
    """執行參數優化"""
    # 載入數據 (與單筆回測相同)
    available_dates = manager.get_available_dates(symbol)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1

    need_download = any(
        (start + timedelta(days=i)).strftime("%Y-%m-%d") not in available_dates
        for i in range(days)
    )

    if need_download:
        with st.spinner("下載歷史數據中..."):
            manager.download_data(symbol, ccxt_symbol, start_date, end_date)

    with st.spinner("載入數據..."):
        df = manager.load_data(symbol, start_date, end_date)

    if df is None or df.empty:
        st.error("載入數據失敗")
        return None

    st.success(f"載入 {len(df):,} 條 K 線")

    # 執行優化
    progress_bar = st.progress(0, text="優化中...")

    def update_progress(current, total):
        progress_bar.progress(current / total, text=f"優化中... {current}/{total}")

    results = manager.optimize_params(sym_config, df, update_progress)

    progress_bar.progress(1.0, text="優化完成!")

    return results


def render_optimization_results(results: list, symbol: str):
    """渲染優化結果"""
    st.subheader("🏆 優化結果 (Top 10)")

    if not results:
        st.warning("無優化結果")
        return

    # 轉換為 DataFrame
    rows = []
    for r in results[:10]:
        rows.append({
            "排名": len(rows) + 1,
            "止盈%": f"{r['take_profit_spacing']*100:.2f}",
            "補倉%": f"{r['grid_spacing']*100:.2f}",
            "收益率%": f"{r['return_pct']*100:.2f}",
            "回撤%": f"{r['max_drawdown']*100:.1f}",
            "勝率%": f"{r['win_rate']*100:.1f}",
            "交易數": r['trades_count'],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 應用最佳參數
    if results:
        best = results[0]
        st.divider()

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(
                f"**最佳參數:** 止盈 {best['take_profit_spacing']*100:.2f}%, "
                f"補倉 {best['grid_spacing']*100:.2f}%"
            )

        with col2:
            if st.button("套用最佳參數", type="primary"):
                config = get_config()
                if symbol not in config.symbols:
                    # 新增
                    config.symbols[symbol] = SymbolConfig(symbol=symbol)

                config.symbols[symbol].take_profit_spacing = best['take_profit_spacing']
                config.symbols[symbol].grid_spacing = best['grid_spacing']
                save_config()

                st.success("已套用最佳參數!")
                st.rerun()


def main():
    """主函數"""
    # 先渲染側邊欄（確保不被 st.stop() 阻擋）
    render_sidebar()

    st.title("🔬 回測 / 優化")
    st.divider()

    manager = get_backtest_manager()

    # 左側：配置
    # 右側：結果
    left, right = st.columns([1, 2])

    with left:
        symbol, ccxt_symbol, sym_config = render_symbol_input()

        if not symbol:
            st.stop()

        st.divider()
        start_date, end_date = render_date_range()

        st.divider()
        sym_config = render_backtest_params(sym_config)

        st.divider()

        # 模式選擇
        mode = st.radio(
            "選擇模式",
            options=["單筆回測", "參數優化"],
            horizontal=True,
        )

        if st.button("🚀 開始", type="primary", use_container_width=True):
            st.session_state.backtest_mode = mode
            st.session_state.backtest_symbol = symbol
            st.session_state.backtest_ccxt = ccxt_symbol
            st.session_state.backtest_config = sym_config
            st.session_state.backtest_start = start_date
            st.session_state.backtest_end = end_date
            st.session_state.run_backtest = True
            st.rerun()

    with right:
        if st.session_state.get("run_backtest"):
            mode = st.session_state.backtest_mode
            symbol = st.session_state.backtest_symbol
            ccxt_symbol = st.session_state.backtest_ccxt
            sym_config = st.session_state.backtest_config
            start_date = st.session_state.backtest_start
            end_date = st.session_state.backtest_end

            if mode == "單筆回測":
                result = run_single_backtest(
                    manager, symbol, ccxt_symbol, sym_config, start_date, end_date
                )
                if result:
                    render_backtest_result(result)
            else:
                results = run_optimization(
                    manager, symbol, ccxt_symbol, sym_config, start_date, end_date
                )
                if results:
                    render_optimization_results(results, symbol)

            st.session_state.run_backtest = False
        else:
            st.info("配置參數後點擊「開始」執行回測")


# 執行頁面
main()
