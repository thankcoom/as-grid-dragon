"""
回測/優化頁面
=============
回測、參數優化、結果分析
支援智能優化 (Optuna TPE/NSGA-II) 與傳統網格搜索
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

# 檢查智能優化是否可用
try:
    from backtest.smart_optimizer import (
        SmartOptimizer, OptimizationObjective, OptimizationMethod,
        TradingMode, MODE_INFO
    )
    from backtest.config import Config as BacktestConfig
    SMART_OPTIMIZER_AVAILABLE = True
except ImportError:
    SMART_OPTIMIZER_AVAILABLE = False
    TradingMode = None
    MODE_INFO = None

init_session_state()

# 支援的交易所列表
SUPPORTED_EXCHANGES = {
    "binance": "幣安 Binance",
    "bybit": "Bybit",
    "bitget": "Bitget",
    "gate": "Gate.io",
}


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


def render_exchange_selector():
    """渲染交易所選擇器"""
    st.subheader("🏦 數據來源")

    config = get_config()
    default_exchange = config.exchange_type

    # 提示說明
    st.caption("💡 CCXT 下載歷史數據不需要 API Key，可以自由選擇交易所")

    col1, col2 = st.columns([2, 1])

    with col1:
        # 找到預設交易所的索引
        exchange_keys = list(SUPPORTED_EXCHANGES.keys())
        default_idx = exchange_keys.index(default_exchange) if default_exchange in exchange_keys else 0

        selected_exchange = st.selectbox(
            "選擇交易所",
            options=exchange_keys,
            format_func=lambda x: SUPPORTED_EXCHANGES[x],
            index=default_idx,
            key="backtest_exchange",
            help="選擇要從哪個交易所下載歷史數據"
        )

    with col2:
        # 顯示當前配置的交易所（用於交易）
        st.caption(f"交易配置: {SUPPORTED_EXCHANGES.get(default_exchange, default_exchange)}")

    return selected_exchange


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
                        sym_config: SymbolConfig, start_date: str, end_date: str,
                        exchange_type: str = "binance"):
    """執行單筆回測"""
    # 顯示使用的交易所
    st.info(f"📡 數據來源: **{SUPPORTED_EXCHANGES.get(exchange_type, exchange_type)}**")

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
            st.info(f"從 {exchange_type.upper()} 下載歷史數據中...")
            manager.download_data(symbol, ccxt_symbol, start_date, end_date, exchange_type)

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
                     sym_config: SymbolConfig, start_date: str, end_date: str,
                     use_smart: bool = True, n_trials: int = 100,
                     objective: str = "sharpe", trading_mode=None,
                     exchange_type: str = "binance"):
    """執行參數優化 - 支援智能優化與傳統網格搜索"""
    # 顯示使用的交易所
    st.info(f"📡 數據來源: **{SUPPORTED_EXCHANGES.get(exchange_type, exchange_type)}**")

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
        with st.spinner(f"從 {exchange_type.upper()} 下載歷史數據中..."):
            manager.download_data(symbol, ccxt_symbol, start_date, end_date, exchange_type)

    with st.spinner("載入數據..."):
        df = manager.load_data(symbol, start_date, end_date)

    if df is None or df.empty:
        st.error("載入數據失敗")
        return None, None, None, None

    st.success(f"載入 {len(df):,} 條 K 線")

    # 智能優化模式
    if use_smart and SMART_OPTIMIZER_AVAILABLE:
        results, smart_result, optimizer = run_smart_optimization(
            df, sym_config, n_trials, objective, trading_mode
        )
        return results, smart_result, optimizer, df
    else:
        # 傳統網格優化
        if use_smart and not SMART_OPTIMIZER_AVAILABLE:
            st.warning("⚠️ 智能優化不可用 (請安裝 Optuna: pip install optuna)，改用傳統網格優化")
        
        progress_bar = st.progress(0, text="網格優化中...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"網格優化中... {current}/{total}")

        results = manager.optimize_params(sym_config, df, update_progress)
        progress_bar.progress(1.0, text="優化完成!")
        
        return results, None, None, df


def run_smart_optimization(df: pd.DataFrame, sym_config: SymbolConfig,
                           n_trials: int, objective: str, trading_mode=None):
    """執行智能優化 (使用 Optuna TPE)"""
    # 轉換配置
    base_config = BacktestConfig(
        symbol=sym_config.symbol,
        initial_quantity=sym_config.initial_quantity,
        leverage=sym_config.leverage,
        take_profit_spacing=sym_config.take_profit_spacing,
        grid_spacing=sym_config.grid_spacing,
    )

    # 選擇優化目標
    objective_map = {
        "return": OptimizationObjective.RETURN,
        "sharpe": OptimizationObjective.SHARPE,
        "sortino": OptimizationObjective.SORTINO,
        "calmar": OptimizationObjective.CALMAR,
        "profit_factor": OptimizationObjective.PROFIT_FACTOR,
        "risk_adjusted": OptimizationObjective.RISK_ADJUSTED,
    }
    opt_objective = objective_map.get(objective, OptimizationObjective.SHARPE)

    # 創建優化器（傳入交易模式）
    optimizer = SmartOptimizer(df, base_config, trading_mode=trading_mode)

    # 顯示使用的模式
    if trading_mode is not None:
        mode_info = MODE_INFO[trading_mode]
        st.info(f"🎯 交易模式: {mode_info['name']} | {mode_info['description']}")
    
    progress_bar = st.progress(0, text="智能優化中...")
    status_text = st.empty()
    
    def update_progress(current, total, best_value):
        progress_bar.progress(current / total, text=f"智能優化中... {current}/{total}")
        status_text.caption(f"當前最佳值: {best_value:.4f}")
    
    # 執行優化
    result = optimizer.optimize(
        n_trials=n_trials,
        objective=opt_objective,
        method=OptimizationMethod.TPE,
        progress_callback=update_progress,
        show_progress=False
    )
    
    progress_bar.progress(1.0, text="智能優化完成!")
    status_text.empty()
    
    # 轉換結果格式以兼容現有顯示
    results = []
    for trial in result.all_trials:
        results.append({
            "take_profit_spacing": trial.params.get("take_profit_spacing", sym_config.take_profit_spacing),
            "grid_spacing": trial.params.get("grid_spacing", sym_config.grid_spacing),
            "limit_multiplier": trial.params.get("limit_multiplier", 5.0),
            "threshold_multiplier": trial.params.get("threshold_multiplier", 14.0),
            "return_pct": trial.metrics.get("return_pct", 0),
            "max_drawdown": trial.metrics.get("max_drawdown", 0),
            "win_rate": trial.metrics.get("win_rate", 0),
            "trades_count": trial.metrics.get("trades_count", 0),
            "sharpe_ratio": trial.metrics.get("sharpe_ratio", 0),
            "objective_value": trial.objective_value,
        })
    
    # 按收益率排序
    results.sort(key=lambda x: x["return_pct"], reverse=True)
    
    # 返回結果、SmartOptimizationResult 和 optimizer（用於獲取 study）
    return results, result, optimizer


def render_optimization_results(results: list, symbol: str, smart_result=None, optimizer=None, 
                                df=None, sym_config=None):
    """渲染優化結果"""
    st.subheader("🏆 優化結果 (Top 10)")

    if not results:
        st.warning("無優化結果")
        return

    # 顯示優化摘要（如果是智能優化）
    if smart_result is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總試驗數", smart_result.n_trials)
        with col2:
            st.metric("優化耗時", f"{smart_result.optimization_time:.1f}s")
        with col3:
            st.metric("最佳目標值", f"{smart_result.best_objective:.4f}")
        with col4:
            st.metric("優化方法", smart_result.method.upper())
        st.divider()

    # 轉換為 DataFrame
    rows = []
    for r in results[:10]:
        row = {
            "排名": len(rows) + 1,
            "止盈%": f"{r['take_profit_spacing']*100:.2f}",
            "補倉%": f"{r['grid_spacing']*100:.2f}",
            "收益率%": f"{r['return_pct']*100:.2f}",
            "回撤%": f"{r['max_drawdown']*100:.1f}",
            "勝率%": f"{r['win_rate']*100:.1f}",
            "交易數": r['trades_count'],
        }
        # 智能優化額外顯示 Sharpe
        if "sharpe_ratio" in r and r["sharpe_ratio"]:
            row["Sharpe"] = f"{r['sharpe_ratio']:.2f}"
        # 顯示新參數（如果被優化）
        if "limit_multiplier" in r:
            row["加倍倍數"] = f"{r['limit_multiplier']:.1f}"
        if "threshold_multiplier" in r:
            row["裝死倍數"] = f"{r['threshold_multiplier']:.1f}"
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # 顯示參數重要性（智能優化）
    if smart_result and smart_result.param_importance:
        st.divider()
        st.markdown("**📊 參數重要性分析**")
        
        import plotly.express as px
        importance_df = pd.DataFrame([
            {"參數": k, "重要性": v}
            for k, v in smart_result.param_importance.items()
        ]).sort_values("重要性", ascending=True)
        
        fig = px.bar(importance_df, x="重要性", y="參數", orientation="h",
                     color="重要性", color_continuous_scale="Blues")
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 智能優化進階視覺化（需要 optimizer 對象）
    if optimizer is not None and SMART_OPTIMIZER_AVAILABLE:
        render_advanced_visualizations(optimizer, smart_result, df, sym_config)

    # 應用最佳參數
    if results:
        best = results[0]
        st.divider()

        col1, col2 = st.columns([3, 1])

        with col1:
            params_str = f"**最佳參數:** 止盈 {best['take_profit_spacing']*100:.2f}%, 補倉 {best['grid_spacing']*100:.2f}%"
            if "limit_multiplier" in best:
                params_str += f", 加倍 {best['limit_multiplier']:.1f}x"
            if "threshold_multiplier" in best:
                params_str += f", 裝死 {best['threshold_multiplier']:.1f}x"
            st.markdown(params_str)

        with col2:
            if st.button("套用最佳參數", type="primary"):
                config = get_config()
                if symbol not in config.symbols:
                    # 新增
                    config.symbols[symbol] = SymbolConfig(symbol=symbol)

                config.symbols[symbol].take_profit_spacing = best['take_profit_spacing']
                config.symbols[symbol].grid_spacing = best['grid_spacing']
                if "limit_multiplier" in best:
                    config.symbols[symbol].limit_multiplier = best['limit_multiplier']
                if "threshold_multiplier" in best:
                    config.symbols[symbol].threshold_multiplier = best['threshold_multiplier']
                save_config()

                st.success("已套用最佳參數!")
                st.rerun()


def render_advanced_visualizations(optimizer, smart_result, df=None, sym_config=None):
    """渲染進階優化視覺化圖表"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.divider()
    st.markdown("### 📈 進階優化分析")
    
    # 獲取 Optuna study 對象
    study = optimizer.get_study()
    if study is None:
        st.warning("無法獲取優化歷史數據")
        return
    
    # 使用 tabs 組織不同的視覺化
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 參數熱力圖", 
        "📉 收斂曲線", 
        "📊 平行座標圖",
        "🎲 蒙特卡羅模擬"
    ])
    
    with tab1:
        render_contour_plot(study, smart_result)
    
    with tab2:
        render_optimization_history(study, smart_result)
    
    with tab3:
        render_parallel_coordinate(study, smart_result)
    
    with tab4:
        render_monte_carlo_simulation(smart_result, df, sym_config)


def render_contour_plot(study, smart_result):
    """渲染參數熱力圖 (Contour Plot)"""
    import plotly.graph_objects as go
    import numpy as np
    
    st.markdown("**參數空間熱力圖**")
    st.caption("顯示兩個參數組合對目標值的影響。寬廣的高值區域表示參數穩健，小範圍高峰可能過擬合。")
    
    try:
        # 從所有試驗中提取數據
        trials_data = []
        for trial in study.trials:
            if trial.state.name == "COMPLETE":
                trials_data.append({
                    "take_profit": trial.params.get("take_profit_spacing", 0) * 100,
                    "grid_spacing": trial.params.get("grid_spacing", 0) * 100,
                    "objective": trial.value
                })
        
        if len(trials_data) < 10:
            st.info("試驗數據不足，無法生成熱力圖 (需要至少 10 個完成的試驗)")
            return
        
        # 轉換為數組
        tp_values = [d["take_profit"] for d in trials_data]
        gs_values = [d["grid_spacing"] for d in trials_data]
        obj_values = [d["objective"] for d in trials_data]
        
        # 創建熱力圖
        fig = go.Figure(data=go.Scatter(
            x=tp_values,
            y=gs_values,
            mode='markers',
            marker=dict(
                size=10,
                color=obj_values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="目標值")
            ),
            text=[f"目標: {v:.4f}" for v in obj_values],
            hovertemplate="止盈: %{x:.2f}%<br>補倉: %{y:.2f}%<br>%{text}<extra></extra>"
        ))
        
        # 標記最佳點
        best_tp = smart_result.best_params.get("take_profit_spacing", 0) * 100
        best_gs = smart_result.best_params.get("grid_spacing", 0) * 100
        
        fig.add_trace(go.Scatter(
            x=[best_tp],
            y=[best_gs],
            mode='markers',
            marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
            name='最佳參數',
            hovertemplate=f"最佳參數<br>止盈: {best_tp:.2f}%<br>補倉: {best_gs:.2f}%<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis_title="止盈間距 (%)",
            yaxis_title="補倉間距 (%)",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 過擬合風險評估
        render_overfitting_assessment(trials_data, smart_result)
        
    except Exception as e:
        st.error(f"生成熱力圖時發生錯誤: {str(e)}")


def render_overfitting_assessment(trials_data, smart_result):
    """評估過擬合風險"""
    import numpy as np
    
    obj_values = [d["objective"] for d in trials_data]
    best_obj = smart_result.best_objective
    
    # 計算統計數據
    mean_obj = np.mean(obj_values)
    std_obj = np.std(obj_values)
    top_10_pct = np.percentile(obj_values, 90)
    
    # 過擬合風險指標
    # 1. 最佳值與平均值的差距（標準差倍數）
    z_score = (best_obj - mean_obj) / std_obj if std_obj > 0 else 0
    
    # 2. 最佳值在 top 10% 中的位置
    top_trials = [d for d in trials_data if d["objective"] >= top_10_pct]
    
    # 3. 計算 top 10% 的參數分散度
    if len(top_trials) > 1:
        tp_std = np.std([d["take_profit"] for d in top_trials])
        gs_std = np.std([d["grid_spacing"] for d in top_trials])
        param_spread = (tp_std + gs_std) / 2
    else:
        param_spread = 0
    
    # 評估風險等級
    if z_score > 3 and param_spread < 0.1:
        risk_level = "⚠️ 高"
        risk_color = "red"
        risk_msg = "最佳參數位於非常狹窄的區域，可能存在過擬合風險。建議使用更長的歷史數據或進行 Walk-Forward 驗證。"
    elif z_score > 2 and param_spread < 0.2:
        risk_level = "🟡 中"
        risk_color = "orange"
        risk_msg = "最佳參數區域較為集中，建議進行樣本外驗證。"
    else:
        risk_level = "✅ 低"
        risk_color = "green"
        risk_msg = "最佳參數位於相對寬廣的區域，參數穩健性較好。"
    
    st.markdown(f"""
    **過擬合風險評估**: <span style="color:{risk_color}">{risk_level}</span>
    
    - Z-Score: {z_score:.2f} (最佳值與平均值的偏離程度)
    - Top 10% 參數分散度: {param_spread:.2f}%
    - {risk_msg}
    """, unsafe_allow_html=True)


def render_optimization_history(study, smart_result):
    """渲染優化收斂曲線"""
    import plotly.graph_objects as go
    
    st.markdown("**優化收斂曲線**")
    st.caption("顯示優化過程中目標值的變化。曲線趨於平穩表示已收斂。")
    
    try:
        # 提取試驗歷史
        trial_numbers = []
        trial_values = []
        best_values = []
        current_best = float('-inf')
        
        for trial in study.trials:
            if trial.state.name == "COMPLETE" and trial.value is not None:
                trial_numbers.append(trial.number + 1)
                trial_values.append(trial.value)
                current_best = max(current_best, trial.value)
                best_values.append(current_best)
        
        if not trial_numbers:
            st.info("無試驗數據可顯示")
            return
        
        fig = go.Figure()
        
        # 所有試驗點
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=trial_values,
            mode='markers',
            name='試驗結果',
            marker=dict(size=6, color='lightblue', opacity=0.6),
            hovertemplate="試驗 #%{x}<br>目標值: %{y:.4f}<extra></extra>"
        ))
        
        # 最佳值曲線
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=best_values,
            mode='lines',
            name='當前最佳',
            line=dict(color='#00CC96', width=3),
            hovertemplate="試驗 #%{x}<br>最佳值: %{y:.4f}<extra></extra>"
        ))
        
        # 標記最終最佳值
        fig.add_hline(y=smart_result.best_objective, line_dash="dash", 
                      line_color="gold", annotation_text=f"最佳: {smart_result.best_objective:.4f}")
        
        fig.update_layout(
            xaxis_title="試驗次數",
            yaxis_title="目標值",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 收斂分析
        if len(best_values) >= 10:
            # 檢查最後 20% 的試驗是否有改善
            cutoff = int(len(best_values) * 0.8)
            early_best = best_values[cutoff] if cutoff < len(best_values) else best_values[-1]
            improvement = (smart_result.best_objective - early_best) / abs(early_best) * 100 if early_best != 0 else 0
            
            if improvement < 1:
                st.success(f"✅ 優化已收斂：最後 20% 試驗改善幅度僅 {improvement:.2f}%")
            else:
                st.warning(f"⚠️ 優化可能未完全收斂：最後 20% 試驗仍有 {improvement:.2f}% 改善，建議增加試驗次數")
        
    except Exception as e:
        st.error(f"生成收斂曲線時發生錯誤: {str(e)}")


def render_parallel_coordinate(study, smart_result):
    """渲染平行座標圖"""
    import plotly.express as px
    import pandas as pd

    st.markdown("**平行座標圖**")
    st.caption("同時顯示所有參數與目標值的關係。追蹤高目標值的線條可以看出參數偏好。")

    try:
        # 從試驗中提取數據
        data = []
        for trial in study.trials:
            if trial.state.name == "COMPLETE" and trial.value is not None:
                row = {
                    "止盈%": trial.params.get("take_profit_spacing", 0) * 100,
                    "補倉%": trial.params.get("grid_spacing", 0) * 100,
                    "加倍倍數": trial.params.get("limit_multiplier", 5.0),
                    "裝死倍數": trial.params.get("threshold_multiplier", 14.0),
                    "目標值": trial.value
                }
                data.append(row)

        if len(data) < 5:
            st.info("試驗數據不足，無法生成平行座標圖")
            return

        df = pd.DataFrame(data)

        # 創建平行座標圖
        fig = px.parallel_coordinates(
            df,
            dimensions=["止盈%", "補倉%", "加倍倍數", "裝死倍數", "目標值"],
            color="目標值",
            color_continuous_scale="RdYlGn",
            labels={"color": "目標值"}
        )

        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=30, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)

        # 參數相關性提示
        st.markdown("""
        **解讀提示**：
        - 觀察顏色較深（目標值高）的線條集中在哪個區間
        - **加倍倍數**: 觸發止盈加倍的倉位閾值倍數（越小越早加倍出貨）
        - **裝死倍數**: 觸發裝死模式的倉位閾值倍數（越小越早停止補倉）
        - 線條交叉較多的區域表示參數之間存在交互作用
        """)
        
    except Exception as e:
        st.error(f"生成平行座標圖時發生錯誤: {str(e)}")


def render_monte_carlo_simulation(smart_result, df=None, sym_config=None):
    """渲染蒙特卡羅模擬分析"""
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    
    st.markdown("**蒙特卡羅模擬**")
    st.caption("使用最佳參數在多個隨機時間窗口進行回測，評估策略穩健性。結果分布越集中，策略越穩健。")
    
    # 優先使用 session state 中的數據（解決按鈕點擊後數據丟失問題）
    if df is None:
        df = st.session_state.get("opt_df")
    if sym_config is None:
        sym_config = st.session_state.get("opt_sym_config")
    if smart_result is None:
        smart_result = st.session_state.get("opt_smart_result")
    
    if df is None or sym_config is None or smart_result is None:
        st.warning("⚠️ 請先執行智能優化，才能進行蒙特卡羅模擬。")
        return
    
    # 顯示數據信息
    st.info(f"📊 可用數據：{len(df):,} 條 K 線")
    
    # 模擬設定
    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.select_slider(
            "模擬次數",
            options=[20, 50, 100, 200],
            value=50,
            key="mc_simulations"
        )
    with col2:
        window_pct = st.select_slider(
            "窗口大小 (%)",
            options=[30, 50, 70, 80],
            value=50,
            key="mc_window",
            help="每次模擬使用的數據比例"
        )
    
    # 顯示已有結果或執行按鈕
    if st.session_state.get("mc_results") is not None:
        # 已有結果，顯示結果和重新執行按鈕
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 重新模擬", key="rerun_mc"):
                st.session_state.mc_results = None
                st.rerun()
        
        render_monte_carlo_results(st.session_state.mc_results, smart_result)
    else:
        # 沒有結果，顯示執行按鈕
        if st.button("🎲 執行蒙特卡羅模擬", key="run_mc", type="primary"):
            results = run_monte_carlo(smart_result, df, sym_config, n_simulations, window_pct)
            if results:
                st.session_state.mc_results = results
                st.rerun()


def run_monte_carlo(smart_result, df, sym_config, n_simulations, window_pct):
    """執行蒙特卡羅模擬，返回結果列表"""
    import numpy as np
    from backtest.backtester import GridBacktester
    from backtest.config import Config as BacktestConfig

    # 獲取優化後的 multiplier 參數
    limit_mult = smart_result.best_params.get("limit_multiplier", 5.0)
    threshold_mult = smart_result.best_params.get("threshold_multiplier", 14.0)

    # 計算 position_limit 和 position_threshold
    initial_qty = sym_config.initial_quantity
    if initial_qty <= 0:
        mid_price = df['close'].median() if len(df) > 0 else 1.0
        initial_qty = 10.0 / mid_price  # 預設 10 USDT

    position_limit = initial_qty * limit_mult
    position_threshold = initial_qty * threshold_mult

    # 準備最佳參數配置
    best_config = BacktestConfig(
        symbol=sym_config.symbol,
        initial_quantity=sym_config.initial_quantity,
        leverage=sym_config.leverage,  # 槓桿使用原始設定
        take_profit_spacing=smart_result.best_params.get("take_profit_spacing", sym_config.take_profit_spacing),
        grid_spacing=smart_result.best_params.get("grid_spacing", sym_config.grid_spacing),
        limit_multiplier=limit_mult,
        threshold_multiplier=threshold_mult,
        position_limit=position_limit,
        position_threshold=position_threshold,
    )
    
    # 計算窗口大小
    total_rows = len(df)
    window_size = int(total_rows * window_pct / 100)
    
    if window_size < 1000:
        st.warning(f"數據量不足，窗口大小 ({window_size} 條) 太小，建議使用更長的歷史數據")
        return None
    
    # 進度條
    progress_bar = st.progress(0, text="蒙特卡羅模擬中...")
    status_text = st.empty()
    
    # 執行模擬
    results = []

    for i in range(n_simulations):
        # 隨機選擇起始點
        max_start = total_rows - window_size
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)

        # 截取數據窗口
        window_df = df.iloc[start_idx:start_idx + window_size].copy().reset_index(drop=True)

        try:
            # 每次模擬創建新的回測器實例
            backtester = GridBacktester(window_df, best_config)
            bt_result = backtester.run()
            results.append({
                "simulation": i + 1,
                "start_idx": start_idx,
                "return_pct": bt_result.return_pct * 100,
                "max_drawdown": bt_result.max_drawdown * 100,
                "win_rate": bt_result.win_rate * 100,
                "trades": bt_result.trades_count,
                "sharpe": bt_result.sharpe_ratio if bt_result.sharpe_ratio else 0,
            })
        except Exception as e:
            # 跳過失敗的模擬
            status_text.caption(f"模擬 {i+1} 失敗: {str(e)[:50]}")
        
        progress_bar.progress((i + 1) / n_simulations, text=f"蒙特卡羅模擬中... {i+1}/{n_simulations}")
    
    progress_bar.progress(1.0, text="✅ 模擬完成!")
    status_text.empty()
    
    if not results:
        st.error("所有模擬都失敗了，請檢查數據或參數")
        return None
    
    return results


def render_monte_carlo_results(results, smart_result):
    """渲染蒙特卡羅模擬結果"""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    df_results = pd.DataFrame(results)
    
    # 統計摘要
    st.markdown("#### 📊 模擬結果統計")
    
    returns = df_results["return_pct"].values
    drawdowns = df_results["max_drawdown"].values
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        st.metric(
            "平均收益率",
            f"{mean_return:.2f}%",
            delta=f"±{std_return:.2f}%"
        )
    
    with col2:
        median_return = np.median(returns)
        st.metric("中位數收益率", f"{median_return:.2f}%")
    
    with col3:
        win_ratio = np.sum(returns > 0) / len(returns) * 100
        st.metric("正收益比例", f"{win_ratio:.1f}%")
    
    with col4:
        worst_case = np.percentile(returns, 5)
        st.metric("5% VaR", f"{worst_case:.2f}%", help="最差 5% 情況的收益率")
    
    # 收益率分布圖
    st.markdown("#### 📈 收益率分布")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("收益率分布", "收益率 vs 最大回撤"),
        horizontal_spacing=0.1
    )
    
    # 直方圖
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=20,
            name="收益率分布",
            marker_color='#636EFA',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 添加平均值線
    fig.add_vline(
        x=np.mean(returns), 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"平均: {np.mean(returns):.2f}%",
        row=1, col=1
    )
    
    # 添加原始回測結果線（如果有）
    original_return = smart_result.best_metrics.get("return_pct", 0) * 100
    if original_return:
        fig.add_vline(
            x=original_return,
            line_dash="dot",
            line_color="gold",
            annotation_text=f"原始: {original_return:.2f}%",
            row=1, col=1
        )
    
    # 散點圖：收益率 vs 回撤
    fig.add_trace(
        go.Scatter(
            x=drawdowns,
            y=returns,
            mode='markers',
            marker=dict(
                size=8,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="收益率%", x=1.02)
            ),
            name="模擬結果",
            hovertemplate="回撤: %{x:.2f}%<br>收益: %{y:.2f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="頻率", row=1, col=1)
    fig.update_xaxes(title_text="最大回撤 (%)", row=1, col=2)
    fig.update_yaxes(title_text="收益率 (%)", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 穩健性評估
    st.markdown("#### 🎯 穩健性評估")
    
    # 計算穩健性指標
    cv = std_return / abs(mean_return) if mean_return != 0 else float('inf')  # 變異係數
    sharpe_consistency = np.mean(df_results["sharpe"].values > 0) * 100  # Sharpe > 0 的比例
    
    # 評估等級
    if cv < 0.5 and win_ratio > 70:
        robustness_level = "✅ 高"
        robustness_color = "green"
        robustness_msg = "策略在不同時間段表現穩定，過擬合風險較低。"
    elif cv < 1.0 and win_ratio > 50:
        robustness_level = "🟡 中"
        robustness_color = "orange"
        robustness_msg = "策略有一定穩健性，但在某些時間段可能表現不佳。"
    else:
        robustness_level = "⚠️ 低"
        robustness_color = "red"
        robustness_msg = "策略表現波動較大，可能存在過擬合風險，建議謹慎使用。"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("變異係數 (CV)", f"{cv:.2f}", help="越低越穩定，< 0.5 為佳")
    
    with col2:
        st.metric("Sharpe > 0 比例", f"{sharpe_consistency:.1f}%")
    
    with col3:
        st.metric("穩健性等級", robustness_level)
    
    st.markdown(f"""
    <div style="padding: 10px; border-left: 4px solid {robustness_color}; background-color: rgba(0,0,0,0.05);">
    <strong>評估結論</strong>: {robustness_msg}
    </div>
    """, unsafe_allow_html=True)
    
    # 詳細數據表格（可展開）
    with st.expander("📋 查看詳細模擬數據"):
        display_df = df_results[["simulation", "return_pct", "max_drawdown", "win_rate", "trades", "sharpe"]].copy()
        display_df.columns = ["模擬#", "收益率%", "最大回撤%", "勝率%", "交易數", "Sharpe"]
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_optimization_settings():
    """渲染優化設定"""
    st.subheader("🧠 優化設定")

    # 優化模式
    use_smart = st.toggle(
        "啟用智能優化 (TPE)",
        value=SMART_OPTIMIZER_AVAILABLE,
        disabled=not SMART_OPTIMIZER_AVAILABLE,
        help="使用 Optuna TPE 算法進行智能參數搜索，比網格搜索更高效"
    )

    if not SMART_OPTIMIZER_AVAILABLE:
        st.caption("⚠️ 請安裝 Optuna: `pip install optuna`")

    if use_smart and SMART_OPTIMIZER_AVAILABLE:
        # 交易模式選擇
        st.markdown("**📋 交易模式**")

        trading_mode_options = [
            TradingMode.HIGH_FREQ,
            TradingMode.SWING,
            TradingMode.LONG_CYCLE,
        ]

        selected_mode = st.radio(
            "選擇交易模式",
            options=trading_mode_options,
            format_func=lambda m: f"{MODE_INFO[m]['name']} ({MODE_INFO[m]['timeframe']})",
            horizontal=True,
            help="不同模式有不同的參數範圍，適合不同的持倉週期"
        )

        # 顯示模式說明
        mode_info = MODE_INFO[selected_mode]
        st.caption(f"💡 {mode_info['description']} | 適合: {mode_info['best_for']}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            n_trials = st.select_slider(
                "試驗次數",
                options=[50, 100, 200, 500],
                value=100,
                help="更多試驗可能找到更好的參數，但耗時更長"
            )

        with col2:
            objective = st.selectbox(
                "優化目標",
                options=["sharpe", "return", "sortino", "calmar", "risk_adjusted"],
                format_func=lambda x: {
                    "return": "📈 收益率 (Return)",
                    "sharpe": "⚖️ 夏普比率 (Sharpe)",
                    "sortino": "📉 索提諾比率 (Sortino)",
                    "calmar": "🛡️ 卡瑪比率 (Calmar)",
                    "risk_adjusted": "🎯 風險調整收益",
                }.get(x, x),
                help="Sharpe: 風險調整收益 | Sortino: 只計算下行風險 | Calmar: 收益/最大回撤"
            )

        return use_smart, n_trials, objective, selected_mode
    else:
        st.info("傳統網格優化: 21 種參數組合")
        return False, 21, "return", None


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
        selected_exchange = render_exchange_selector()

        st.divider()
        sym_config = render_backtest_params(sym_config)

        st.divider()

        # 模式選擇
        mode = st.radio(
            "選擇模式",
            options=["單筆回測", "參數優化"],
            horizontal=True,
        )
        
        # 優化設定（僅在參數優化模式顯示）
        use_smart, n_trials, objective, trading_mode = False, 21, "return", None
        if mode == "參數優化":
            st.divider()
            use_smart, n_trials, objective, trading_mode = render_optimization_settings()

        st.divider()

        if st.button("🚀 開始", type="primary", use_container_width=True):
            st.session_state.backtest_mode = mode
            st.session_state.backtest_symbol = symbol
            st.session_state.backtest_ccxt = ccxt_symbol
            st.session_state.backtest_config = sym_config
            st.session_state.backtest_start = start_date
            st.session_state.backtest_end = end_date
            st.session_state.backtest_exchange = selected_exchange
            st.session_state.use_smart = use_smart
            st.session_state.n_trials = n_trials
            st.session_state.objective = objective
            st.session_state.trading_mode = trading_mode
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
            exchange_type = st.session_state.get("backtest_exchange", "binance")

            if mode == "單筆回測":
                result = run_single_backtest(
                    manager, symbol, ccxt_symbol, sym_config, start_date, end_date,
                    exchange_type=exchange_type
                )
                if result:
                    render_backtest_result(result)
            else:
                use_smart = st.session_state.get("use_smart", False)
                n_trials = st.session_state.get("n_trials", 100)
                objective = st.session_state.get("objective", "sharpe")
                trading_mode = st.session_state.get("trading_mode", None)

                results, smart_result, optimizer, opt_df = run_optimization(
                    manager, symbol, ccxt_symbol, sym_config, start_date, end_date,
                    use_smart=use_smart, n_trials=n_trials, objective=objective,
                    trading_mode=trading_mode, exchange_type=exchange_type
                )
                if results:
                    # 保存到 session state 供蒙特卡羅模擬使用
                    st.session_state.opt_results = results
                    st.session_state.opt_smart_result = smart_result
                    st.session_state.opt_optimizer = optimizer
                    st.session_state.opt_df = opt_df
                    st.session_state.opt_sym_config = sym_config
                    st.session_state.opt_symbol = symbol
                    
                    render_optimization_results(
                        results, symbol, smart_result, optimizer, 
                        df=opt_df, sym_config=sym_config
                    )

            st.session_state.run_backtest = False
        elif st.session_state.get("opt_results") is not None:
            # 已有優化結果，從 session state 恢復顯示
            results = st.session_state.opt_results
            symbol = st.session_state.opt_symbol
            smart_result = st.session_state.opt_smart_result
            optimizer = st.session_state.opt_optimizer
            opt_df = st.session_state.opt_df
            sym_config = st.session_state.opt_sym_config
            
            render_optimization_results(
                results, symbol, smart_result, optimizer,
                df=opt_df, sym_config=sym_config
            )
        else:
            st.info("配置參數後點擊「開始」執行回測")


# 執行頁面
main()
