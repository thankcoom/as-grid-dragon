"""
交易對管理頁面
==============
新增、編輯、刪除交易對
"""

import streamlit as st

st.set_page_config(
    page_title="交易對管理 - AS 網格",
    page_icon="⚙️",
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

init_session_state()

# 交易模式定義
TRADING_MODES = {
    "high_freq": {"name": "🚀 次高頻", "desc": "2-7天，小間距"},
    "swing": {"name": "📊 波動", "desc": "1週-1月，中間距"},
    "long_cycle": {"name": "🌊 大週期", "desc": "1月以上，大間距"},
}


def render_symbols_list():
    """渲染交易對列表"""
    st.subheader("📋 已配置交易對")

    config = get_config()

    if not config.symbols:
        st.info("尚未配置任何交易對")
        return

    for symbol, cfg in list(config.symbols.items()):
        status_icon = "🟢" if cfg.enabled else "⚪"
        mode_key = getattr(cfg, 'trading_mode', 'swing')
        mode_info = TRADING_MODES.get(mode_key, TRADING_MODES['swing'])

        with st.expander(f"{status_icon} {symbol}  {mode_info['name']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**止盈:** {cfg.take_profit_spacing*100:.2f}%")
                st.write(f"**補倉:** {cfg.grid_spacing*100:.2f}%")

            with col2:
                st.write(f"**數量:** {cfg.initial_quantity}")
                st.write(f"**槓桿:** {cfg.leverage}x")

            with col3:
                st.write(f"**加倍觸發:** {cfg.position_limit:.1f}")
                st.write(f"**裝死觸發:** {cfg.position_threshold:.1f}")

            with col4:
                # 操作按鈕
                c1, c2, c3 = st.columns(3)

                with c1:
                    toggle_label = "停用" if cfg.enabled else "啟用"
                    if st.button(toggle_label, key=f"toggle_{symbol}"):
                        cfg.enabled = not cfg.enabled
                        save_config()
                        st.rerun()

                with c2:
                    if st.button("編輯", key=f"edit_{symbol}"):
                        st.session_state.editing_symbol = symbol
                        st.rerun()

                with c3:
                    if st.button("刪除", key=f"del_{symbol}", type="secondary"):
                        st.session_state.deleting_symbol = symbol


def render_add_symbol():
    """渲染新增交易對表單"""
    st.subheader("➕ 新增交易對")

    with st.form("add_symbol_form"):
        symbol_input = st.text_input(
            "交易對名稱",
            placeholder="例如: XRPUSDC, BTC/USDT, ethusdt",
            help="支援多種格式，系統會自動識別"
        )

        col1, col2 = st.columns(2)

        with col1:
            take_profit = st.number_input(
                "止盈間距 (%)",
                min_value=0.1,
                max_value=10.0,
                value=0.4,
                step=0.1,
                help="達到此漲幅後止盈"
            )

            grid_spacing = st.number_input(
                "補倉間距 (%)",
                min_value=0.1,
                max_value=10.0,
                value=0.6,
                step=0.1,
                help="下跌此幅度後補倉"
            )

        with col2:
            quantity = st.number_input(
                "每單數量",
                min_value=1.0,
                max_value=10000.0,
                value=10.0,
                step=1.0,
                help="每次開倉的數量"
            )

            leverage = st.number_input(
                "槓桿倍數",
                min_value=1,
                max_value=125,
                value=20,
                step=1,
                help="合約槓桿倍數"
            )

        # 交易模式選擇
        st.markdown("**交易模式**")
        trading_mode = st.radio(
            "選擇交易模式",
            options=list(TRADING_MODES.keys()),
            format_func=lambda m: f"{TRADING_MODES[m]['name']} ({TRADING_MODES[m]['desc']})",
            horizontal=True,
            index=1,  # 預設波動模式
            help="不同模式適合不同的持倉週期"
        )

        # 進階選項
        with st.expander("進階選項"):
            limit_mult = st.number_input(
                "加倍止盈倍數",
                min_value=1.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="持倉達到 (數量 × 此倍數) 時觸發 2x 止盈"
            )

            threshold_mult = st.number_input(
                "裝死模式倍數",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="持倉達到 (數量 × 此倍數) 時進入裝死模式"
            )

        submitted = st.form_submit_button("新增交易對", type="primary")

        if submitted:
            if not symbol_input:
                st.error("請輸入交易對名稱")
                return

            # 解析交易對
            raw, ccxt_sym, coin, quote = normalize_symbol(symbol_input)

            if not raw:
                st.error(f"無法識別交易對格式: {symbol_input}")
                return

            config = get_config()

            if raw in config.symbols:
                st.warning(f"{raw} 已存在")
                return

            # 新增交易對
            config.symbols[raw] = SymbolConfig(
                symbol=raw,
                ccxt_symbol=ccxt_sym,
                enabled=True,
                take_profit_spacing=take_profit / 100,
                grid_spacing=grid_spacing / 100,
                initial_quantity=quantity,
                leverage=leverage,
                limit_multiplier=limit_mult,
                threshold_multiplier=threshold_mult,
                trading_mode=trading_mode,
            )
            save_config()

            st.success(f"已新增 {raw} ({coin}/{quote})")
            st.rerun()


def render_edit_symbol():
    """渲染編輯交易對表單"""
    symbol = st.session_state.get("editing_symbol")
    if not symbol:
        return

    config = get_config()
    if symbol not in config.symbols:
        st.session_state.editing_symbol = None
        st.rerun()
        return

    cfg = config.symbols[symbol]

    st.subheader(f"✏️ 編輯 {symbol}")

    with st.form("edit_symbol_form"):
        col1, col2 = st.columns(2)

        with col1:
            take_profit = st.number_input(
                "止盈間距 (%)",
                min_value=0.1,
                max_value=10.0,
                value=cfg.take_profit_spacing * 100,
                step=0.1,
            )

            grid_spacing = st.number_input(
                "補倉間距 (%)",
                min_value=0.1,
                max_value=10.0,
                value=cfg.grid_spacing * 100,
                step=0.1,
            )

            quantity = st.number_input(
                "每單數量",
                min_value=1.0,
                max_value=10000.0,
                value=float(cfg.initial_quantity),
                step=1.0,
            )

        with col2:
            leverage = st.number_input(
                "槓桿倍數",
                min_value=1,
                max_value=125,
                value=cfg.leverage,
                step=1,
            )

            limit_mult = st.number_input(
                "加倍止盈倍數",
                min_value=1.0,
                max_value=50.0,
                value=float(cfg.limit_multiplier),
                step=1.0,
            )

            threshold_mult = st.number_input(
                "裝死模式倍數",
                min_value=1.0,
                max_value=100.0,
                value=float(cfg.threshold_multiplier),
                step=1.0,
            )

        # 交易模式選擇
        st.markdown("**交易模式**")
        current_mode = getattr(cfg, 'trading_mode', 'swing')
        mode_keys = list(TRADING_MODES.keys())
        current_idx = mode_keys.index(current_mode) if current_mode in mode_keys else 1

        trading_mode = st.radio(
            "選擇交易模式",
            options=mode_keys,
            format_func=lambda m: f"{TRADING_MODES[m]['name']} ({TRADING_MODES[m]['desc']})",
            horizontal=True,
            index=current_idx,
            key="edit_trading_mode"
        )

        # 顯示計算值
        st.caption(f"position_limit = {quantity * limit_mult:.1f}")
        st.caption(f"position_threshold = {quantity * threshold_mult:.1f}")

        c1, c2 = st.columns(2)
        with c1:
            if st.form_submit_button("保存", type="primary"):
                cfg.take_profit_spacing = take_profit / 100
                cfg.grid_spacing = grid_spacing / 100
                cfg.initial_quantity = quantity
                cfg.leverage = leverage
                cfg.limit_multiplier = limit_mult
                cfg.threshold_multiplier = threshold_mult
                cfg.trading_mode = trading_mode
                save_config()

                st.session_state.editing_symbol = None
                st.success("已保存")
                st.rerun()

        with c2:
            if st.form_submit_button("取消"):
                st.session_state.editing_symbol = None
                st.rerun()


def render_delete_confirm():
    """渲染刪除確認"""
    symbol = st.session_state.get("deleting_symbol")
    if not symbol:
        return

    st.warning(f"確定要刪除 {symbol} 嗎？")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("確定刪除", type="primary"):
            config = get_config()
            if symbol in config.symbols:
                del config.symbols[symbol]
                save_config()
            st.session_state.deleting_symbol = None
            st.success(f"已刪除 {symbol}")
            st.rerun()

    with col2:
        if st.button("取消"):
            st.session_state.deleting_symbol = None
            st.rerun()


def main():
    """主函數"""
    # 先渲染側邊欄
    render_sidebar()

    st.title("⚙️ 交易對管理")
    st.divider()

    # 檢查是否有待處理的操作
    if st.session_state.get("deleting_symbol"):
        render_delete_confirm()
        return

    if st.session_state.get("editing_symbol"):
        render_edit_symbol()
        return

    # 正常顯示
    left, right = st.columns([2, 1])

    with left:
        render_symbols_list()

    with right:
        render_add_symbol()


# 執行頁面
main()
