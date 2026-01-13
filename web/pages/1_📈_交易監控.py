"""
交易監控頁面
============
即時持倉、浮盈、成交記錄
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="交易監控 - AS 網格",
    page_icon="📈",
    layout="wide",
)

# 導入狀態管理
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from theme import apply_custom_theme
from components.sidebar import render_sidebar
apply_custom_theme()

from state import (
    init_session_state,
    is_trading_active,
    get_trading_stats,
    get_trading_duration,
    get_bot,
    get_config,
    stop_trading,
)

init_session_state()


def render_header():
    """渲染標題"""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title("📈 交易監控")

    with col2:
        if is_trading_active():
            duration = get_trading_duration()
            st.metric("運行時間", duration or "00:00:00")

    with col3:
        if is_trading_active():
            if st.button("⏹️ 停止", type="secondary"):
                success, msg = stop_trading()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


def render_positions_table():
    """渲染持倉表格"""
    st.subheader("📊 當前持倉")

    if not is_trading_active():
        st.info("交易未運行")
        return

    stats = get_trading_stats()
    symbols = stats.get("symbols", {})

    if not symbols:
        st.info("等待數據...")
        return

    # 轉換為 DataFrame
    rows = []
    for name, data in symbols.items():
        long_pnl = 0
        short_pnl = 0

        # 計算多空浮盈 (簡化計算)
        if data['long_position'] > 0 and data['long_avg_price'] > 0:
            long_pnl = (data['price'] - data['long_avg_price']) * data['long_position']
        if data['short_position'] > 0 and data['short_avg_price'] > 0:
            short_pnl = (data['short_avg_price'] - data['price']) * data['short_position']

        rows.append({
            "交易對": name,
            "當前價格": f"{data['price']:.6f}",
            "多單持倉": f"{data['long_position']:.2f}",
            "多單均價": f"{data['long_avg_price']:.6f}" if data['long_avg_price'] > 0 else "-",
            "空單持倉": f"{data['short_position']:.2f}",
            "空單均價": f"{data['short_avg_price']:.6f}" if data['short_avg_price'] > 0 else "-",
            "浮盈(U)": f"{data['unrealized_pnl']:+.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 總計
    total_pnl = sum(d['unrealized_pnl'] for d in symbols.values())
    color = "green" if total_pnl >= 0 else "red"
    st.markdown(f"**總浮盈: :{color}[{total_pnl:+.2f} U]**")


def render_account_balances():
    """渲染帳戶餘額"""
    st.subheader("💰 帳戶餘額")

    if not is_trading_active():
        st.info("交易未運行")
        return

    stats = get_trading_stats()
    accounts = stats.get("accounts", {})

    if not accounts:
        st.info("等待數據...")
        return

    cols = st.columns(len(accounts))
    for i, (name, acc) in enumerate(accounts.items()):
        with cols[i]:
            st.metric(
                name,
                f"{acc['wallet_balance']:.2f}",
                delta=f"可用: {acc['available_balance']:.2f}",
                delta_color="off"
            )


def render_recent_trades():
    """渲染最近成交"""
    st.subheader("📝 最近成交")

    bot = get_bot()
    if not bot or not is_trading_active():
        st.info("交易未運行")
        return

    # 收集所有交易對的最近成交
    all_trades = []
    for name, sym_state in bot.state.symbols.items():
        for trade in sym_state.recent_trades:
            all_trades.append({
                "時間": trade.get("time", "-"),
                "交易對": name,
                "方向": trade.get("side", "-"),
                "數量": trade.get("qty", 0),
                "價格": trade.get("price", 0),
                "盈虧": trade.get("pnl", 0),
            })

    if not all_trades:
        st.info("暫無成交記錄")
        return

    # 按時間排序
    all_trades.sort(key=lambda x: x["時間"], reverse=True)

    # 顯示最近 20 條
    df = pd.DataFrame(all_trades[:20])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_symbol_details():
    """渲染交易對詳情"""
    st.subheader("🔍 交易對詳情")

    config = get_config()
    if not config.symbols:
        st.info("未配置交易對")
        return

    # 選擇交易對
    symbol = st.selectbox(
        "選擇交易對",
        options=list(config.symbols.keys()),
        key="detail_symbol"
    )

    if not symbol:
        return

    cfg = config.symbols[symbol]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**策略參數**")
        st.write(f"- 止盈間距: {cfg.take_profit_spacing*100:.2f}%")
        st.write(f"- 補倉間距: {cfg.grid_spacing*100:.2f}%")
        st.write(f"- 每單數量: {cfg.initial_quantity}")
        st.write(f"- 槓桿: {cfg.leverage}x")

    with col2:
        st.markdown("**倉位控制**")
        st.write(f"- 加倍止盈觸發: {cfg.position_limit:.1f}")
        st.write(f"- 裝死模式觸發: {cfg.position_threshold:.1f}")
        st.write(f"- 加倍倍數: {cfg.limit_multiplier}x")
        st.write(f"- 裝死倍數: {cfg.threshold_multiplier}x")

    # 如果交易中，顯示即時狀態
    if is_trading_active():
        bot = get_bot()
        if bot and symbol in bot.state.symbols:
            sym_state = bot.state.symbols[symbol]

            st.divider()
            st.markdown("**即時狀態**")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("最新價", f"{sym_state.latest_price:.6f}")
            with c2:
                st.metric("買一價", f"{sym_state.best_bid:.6f}")
            with c3:
                st.metric("賣一價", f"{sym_state.best_ask:.6f}")
            with c4:
                spread = sym_state.best_ask - sym_state.best_bid
                spread_pct = (spread / sym_state.latest_price * 100) if sym_state.latest_price > 0 else 0
                st.metric("價差", f"{spread_pct:.4f}%")

            # 裝死模式狀態
            if sym_state.long_dead_mode or sym_state.short_dead_mode:
                st.warning(
                    f"⚠️ 裝死模式: "
                    f"{'多單' if sym_state.long_dead_mode else ''}"
                    f"{'空單' if sym_state.short_dead_mode else ''}"
                )


def main():
    """主函數"""
    # 先渲染側邊欄
    render_sidebar()

    render_header()
    st.divider()

    # 主要區域
    left, right = st.columns([2, 1])

    with left:
        render_positions_table()
        st.divider()
        render_recent_trades()

    with right:
        render_account_balances()
        st.divider()
        render_symbol_details()

    # 自動刷新
    if is_trading_active():
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, key="monitor_refresh")
        except ImportError:
            st.caption("提示: 安裝 streamlit-autorefresh 可自動刷新")
            if st.button("🔄 手動刷新"):
                st.rerun()


# 執行頁面
main()
