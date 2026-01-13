"""
AS 網格交易系統 - Web UI 主應用
================================
Dashboard 首頁 - 專業交易儀表板風格
"""

import streamlit as st

# 頁面配置 (必須在最前面)
st.set_page_config(
    page_title="Louis Grid",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 導入主題和狀態管理
from theme import apply_custom_theme, render_status_badge, render_metric_card, render_header_with_logo
from components.sidebar import render_sidebar
from state import (
    init_session_state,
    get_config,
    is_trading_active,
    start_trading,
    stop_trading,
    get_trading_duration,
    get_trading_stats,
    get_bot,
)

# 套用自訂主題
apply_custom_theme()

# 初始化
init_session_state()


def render_header():
    """渲染頁面標題"""
    col1, col2 = st.columns([3, 1])

    with col1:
        render_header_with_logo("Louis Grid", "MAX 增強版 網格交易系統")

    with col2:
        if is_trading_active():
            st.markdown(
                render_status_badge("running", "● 交易運行中"),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                render_status_badge("stopped", "○ 未運行"),
                unsafe_allow_html=True
            )


def render_main_metrics():
    """渲染主要指標"""
    config = get_config()

    if is_trading_active():
        stats = get_trading_stats()
        duration = get_trading_duration()
        total_pnl = stats.get('total_unrealized_pnl', 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pnl_color = "#00D68F" if total_pnl >= 0 else "#FF4D4F"
            st.markdown(f"""
            <div class="card">
                <div class="card-header">總浮盈</div>
                <div class="card-value" style="color: {pnl_color};">{total_pnl:+.2f} U</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">運行時間</div>
                <div class="card-value">{duration or "00:00:00"}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            enabled_count = sum(1 for s in config.symbols.values() if s.enabled)
            st.markdown(f"""
            <div class="card">
                <div class="card-header">運行交易對</div>
                <div class="card-value">{enabled_count}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # 帳戶餘額
            accounts = stats.get("accounts", {})
            total_balance = sum(a.get('wallet_balance', 0) for a in accounts.values())
            st.markdown(f"""
            <div class="card">
                <div class="card-header">帳戶餘額</div>
                <div class="card-value">{total_balance:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # 未運行時顯示配置摘要
        enabled = [s for s in config.symbols.values() if s.enabled]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">已配置交易對</div>
                <div class="card-value">{len(config.symbols)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">已啟用</div>
                <div class="card-value">{len(enabled)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            api_status = "✓" if config.api_key else "✗"
            st.markdown(f"""
            <div class="card">
                <div class="card-header">API 狀態</div>
                <div class="card-value">{api_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            mode = "增強" if config.max_enhancement.all_enhancements_enabled else "純淨"
            st.markdown(f"""
            <div class="card">
                <div class="card-header">交易模式</div>
                <div class="card-value">{mode}</div>
            </div>
            """, unsafe_allow_html=True)


def render_control_panel():
    """渲染控制面板"""
    st.markdown("### 控制面板")

    config = get_config()

    if is_trading_active():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.info("🟢 交易系統正在運行，監控所有已啟用的交易對")

        with col2:
            if st.button("📊 查看詳情", use_container_width=True):
                st.switch_page("pages/1_📈_交易監控.py")

        with col3:
            if st.button("⏹️ 停止交易", type="primary", use_container_width=True):
                with st.spinner("正在停止..."):
                    success, msg = stop_trading()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    else:
        # 啟動前檢查
        can_start = True
        warnings = []

        if not config.api_key:
            can_start = False
            warnings.append("⚠️ 請先設定 API Key")

        enabled = [s for s in config.symbols.values() if s.enabled]
        if not enabled:
            can_start = False
            warnings.append("⚠️ 請至少啟用一個交易對")

        if warnings:
            for w in warnings:
                st.warning(w)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if can_start:
                st.info(f"準備就緒，將運行 {len(enabled)} 個交易對")
            else:
                st.info("請完成上述設定後再啟動交易")

        with col2:
            if st.button("⚙️ 設定", use_container_width=True):
                st.switch_page("pages/4_🛠️_設定.py")

        with col3:
            if st.button("▶️ 開始交易", type="primary", use_container_width=True, disabled=not can_start):
                with st.spinner("連接交易所中..."):
                    success, msg = start_trading()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


def render_positions_preview():
    """渲染持倉預覽"""
    st.markdown("### 持倉概覽")

    config = get_config()

    if not config.symbols:
        st.info("尚未配置交易對，請先新增交易對")
        if st.button("➕ 新增交易對"):
            st.switch_page("pages/2_⚙️_交易對管理.py")
        return

    if is_trading_active():
        stats = get_trading_stats()
        symbols_data = stats.get("symbols", {})

        if symbols_data:
            # 建立持倉表格
            for symbol, data in symbols_data.items():
                cfg = config.symbols.get(symbol)
                if not cfg:
                    continue

                pnl = data.get('unrealized_pnl', 0)
                pnl_color = "#00D68F" if pnl >= 0 else "#FF4D4F"

                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1])

                    with col1:
                        st.markdown(f"**{symbol}**")
                        st.caption(f"止盈 {cfg.take_profit_spacing*100:.2f}% | 補倉 {cfg.grid_spacing*100:.2f}%")

                    with col2:
                        st.metric("價格", f"{data['price']:.6f}")

                    with col3:
                        st.metric("多單", f"{data['long_position']:.2f}")

                    with col4:
                        st.metric("空單", f"{data['short_position']:.2f}")

                    with col5:
                        st.markdown(f"""
                        <div style="text-align: right; padding-top: 10px;">
                            <span style="color: {pnl_color}; font-size: 20px; font-weight: 700;">
                                {pnl:+.2f}
                            </span>
                            <br>
                            <span style="color: #8B8D97; font-size: 12px;">浮盈</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()
        else:
            st.info("等待市場數據...")

    else:
        # 顯示配置的交易對 (非運行狀態)
        for symbol, cfg in config.symbols.items():
            status_icon = "🟢" if cfg.enabled else "⚪"

            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"{status_icon} **{symbol}**")

                with col2:
                    st.caption(f"止盈: {cfg.take_profit_spacing*100:.2f}%")

                with col3:
                    st.caption(f"補倉: {cfg.grid_spacing*100:.2f}%")

                with col4:
                    st.caption(f"數量: {cfg.initial_quantity}")

            st.divider()




def main():
    """主函數"""
    render_header()
    st.divider()

    render_main_metrics()
    st.divider()

    render_control_panel()
    st.divider()

    # 持倉概覽 (全寬)
    render_positions_preview()

    # 側邊欄
    render_sidebar()

    # 自動刷新 (交易運行中時)
    if is_trading_active():
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=3000, key="dashboard_refresh")
        except ImportError:
            pass


# 執行頁面
main()
