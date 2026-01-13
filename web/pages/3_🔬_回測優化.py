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
    from backtest.smart_optimizer import SmartOptimizer, OptimizationObjective, OptimizationMethod
    from backtest.config import Config as BacktestConfig
    SMART_OPTIMIZER_AVAILABLE = True
except ImportError:
    SMART_OPTIMIZER_AVAILABLE = False

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
                     sym_config: SymbolConfig, start_date: str, end_date: str,
                     use_smart: bool = True, n_trials: int = 100,
                     objective: str = "sharpe"):
    """執行參數優化 - 支援智能優化與傳統網格搜索"""
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
        return None, None

    st.success(f"載入 {len(df):,} 條 K 線")

    # 智能優化模式
    if use_smart and SMART_OPTIMIZER_AVAILABLE:
        return run_smart_optimization(df, sym_config, n_trials, objective)
    else:
        # 傳統網格優化
        if use_smart and not SMART_OPTIMIZER_AVAILABLE:
            st.warning("⚠️ 智能優化不可用 (請安裝 Optuna: pip install optuna)，改用傳統網格優化")
        
        progress_bar = st.progress(0, text="網格優化中...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"網格優化中... {current}/{total}")

        results = manager.optimize_params(sym_config, df, update_progress)
        progress_bar.progress(1.0, text="優化完成!")
        
        return results, None


def run_smart_optimization(df: pd.DataFrame, sym_config: SymbolConfig, 
                           n_trials: int, objective: str):
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
    
    # 創建優化器
    optimizer = SmartOptimizer(df, base_config)
    
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
            "leverage": trial.params.get("leverage", sym_config.leverage),
            "return_pct": trial.metrics.get("return_pct", 0),
            "max_drawdown": trial.metrics.get("max_drawdown", 0),
            "win_rate": trial.metrics.get("win_rate", 0),
            "trades_count": trial.metrics.get("trades_count", 0),
            "sharpe_ratio": trial.metrics.get("sharpe_ratio", 0),
            "objective_value": trial.objective_value,
        })
    
    # 按收益率排序
    results.sort(key=lambda x: x["return_pct"], reverse=True)
    
    return results, result


def render_optimization_results(results: list, symbol: str, smart_result=None):
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
        # 顯示槓桿（如果被優化）
        if "leverage" in r:
            row["槓桿"] = r["leverage"]
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

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

    # 應用最佳參數
    if results:
        best = results[0]
        st.divider()

        col1, col2 = st.columns([3, 1])

        with col1:
            params_str = f"**最佳參數:** 止盈 {best['take_profit_spacing']*100:.2f}%, 補倉 {best['grid_spacing']*100:.2f}%"
            if "leverage" in best:
                params_str += f", 槓桿 {best['leverage']}x"
            st.markdown(params_str)

        with col2:
            if st.button("套用最佳參數", type="primary"):
                config = get_config()
                if symbol not in config.symbols:
                    # 新增
                    config.symbols[symbol] = SymbolConfig(symbol=symbol)

                config.symbols[symbol].take_profit_spacing = best['take_profit_spacing']
                config.symbols[symbol].grid_spacing = best['grid_spacing']
                if "leverage" in best:
                    config.symbols[symbol].leverage = best['leverage']
                save_config()

                st.success("已套用最佳參數!")
                st.rerun()


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
        
        return use_smart, n_trials, objective
    else:
        st.info("傳統網格優化: 21 種參數組合")
        return False, 21, "return"


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
        
        # 優化設定（僅在參數優化模式顯示）
        use_smart, n_trials, objective = False, 21, "return"
        if mode == "參數優化":
            st.divider()
            use_smart, n_trials, objective = render_optimization_settings()

        st.divider()
        
        if st.button("🚀 開始", type="primary", use_container_width=True):
            st.session_state.backtest_mode = mode
            st.session_state.backtest_symbol = symbol
            st.session_state.backtest_ccxt = ccxt_symbol
            st.session_state.backtest_config = sym_config
            st.session_state.backtest_start = start_date
            st.session_state.backtest_end = end_date
            st.session_state.use_smart = use_smart
            st.session_state.n_trials = n_trials
            st.session_state.objective = objective
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
                use_smart = st.session_state.get("use_smart", False)
                n_trials = st.session_state.get("n_trials", 100)
                objective = st.session_state.get("objective", "sharpe")
                
                results, smart_result = run_optimization(
                    manager, symbol, ccxt_symbol, sym_config, start_date, end_date,
                    use_smart=use_smart, n_trials=n_trials, objective=objective
                )
                if results:
                    render_optimization_results(results, symbol, smart_result)

            st.session_state.run_backtest = False
        else:
            st.info("配置參數後點擊「開始」執行回測")


# 執行頁面
main()
