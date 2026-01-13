"""
設定頁面
========
API、增強功能、學習模組、風控設定
"""

import streamlit as st

st.set_page_config(
    page_title="設定 - AS 網格",
    page_icon="🛠️",
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

from state import init_session_state, get_config, save_config, reload_config

init_session_state()


def render_api_settings():
    """渲染 API 設定"""
    st.subheader("🔑 API 設定")

    config = get_config()

    # 導入交易所列表
    from exchanges import list_supported_exchanges, get_exchange_display_name

    # === 交易所選擇 ===
    st.markdown("**選擇交易所**")

    # 只顯示已支援的交易所
    supported = list_supported_exchanges()
    current_idx = supported.index(config.exchange_type) if config.exchange_type in supported else 0

    selected_exchange = st.selectbox(
        "選擇交易所",
        options=supported,
        format_func=lambda x: get_exchange_display_name(x),
        index=current_idx,
        help="選擇要連接的交易所"
    )

    if selected_exchange != config.exchange_type:
        config.exchange_type = selected_exchange
        # 切換交易所時重置驗證狀態
        st.session_state.api_verified = False
        save_config()
        st.success(f"已切換至 {get_exchange_display_name(selected_exchange)}")
        st.rerun()

    # Testnet 開關
    testnet = st.toggle(
        "使用測試網",
        value=getattr(config, 'testnet', False),
        help="在測試網環境下運行 (不會影響真實資產)"
    )
    if testnet != getattr(config, 'testnet', False):
        config.testnet = testnet
        save_config()
        st.rerun()

    st.divider()

    # === 連線狀態顯示 ===
    api_verified = st.session_state.get("api_verified", False)

    if config.api_key:
        if api_verified:
            st.success(f"✅ API 已驗證 | {get_exchange_display_name(config.exchange_type)} | Key: {config.api_key[:8]}...{config.api_key[-4:]}")
        else:
            st.warning(f"⚠️ API 未驗證 | Key: {config.api_key[:8]}...{config.api_key[-4:]} | 請點擊「驗證並保存」")
    else:
        st.error("❌ 尚未設定 API - 交易功能無法使用")

    with st.expander("修改 API 設定", expanded=not config.api_key):
        api_key = st.text_input(
            "API Key",
            value=config.api_key or "",
            type="password",
        )

        api_secret = st.text_input(
            "API Secret",
            value=config.api_secret or "",
            type="password",
        )

        # Bitget 專用 Passphrase 欄位
        api_password = ""
        if config.exchange_type == "bitget":
            st.info("Bitget 需要額外的 Passphrase（創建 API 時設定的密碼短語）")
            api_password = st.text_input(
                "Passphrase (密碼短語)",
                value=config.api_password or "",
                type="password",
                help="Bitget 官方 API 的三因素認證要求"
            )

        # 驗證並保存按鈕
        if st.button("🔐 驗證並保存 API", type="primary", width='stretch'):
            if not api_key or not api_secret:
                st.error("請先填入 API Key 和 Secret")
            elif config.exchange_type == "bitget" and not api_password:
                st.error("Bitget 需要填入 Passphrase")
            else:
                # 先驗證，驗證成功才保存
                verified = verify_and_save_api(
                    api_key, api_secret, config.exchange_type, api_password
                )
                if verified:
                    config.api_key = api_key
                    config.api_secret = api_secret
                    if config.exchange_type == "bitget":
                        config.api_password = api_password
                    save_config()
                    st.session_state.api_verified = True
                    st.rerun()

        # 僅測試連線（不保存）
        st.caption("或者")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧪 僅測試連線"):
                if not api_key or not api_secret:
                    st.error("請先填入 API Key 和 Secret")
                elif config.exchange_type == "bitget" and not api_password:
                    st.error("Bitget 需要填入 Passphrase")
                else:
                    test_api_connection(api_key, api_secret, config.exchange_type, api_password)

        with col2:
            if st.button("💾 僅保存（跳過驗證）"):
                config.api_key = api_key
                config.api_secret = api_secret
                if config.exchange_type == "bitget":
                    config.api_password = api_password
                st.session_state.api_verified = False  # 標記為未驗證
                save_config()
                st.warning("⚠️ API 已保存但未驗證，建議執行驗證")
                st.rerun()


def verify_and_save_api(api_key: str, api_secret: str, exchange_type: str = "binance", password: str = "") -> bool:
    """驗證 API 連線，成功返回 True"""
    try:
        from exchanges import get_adapter, get_exchange_display_name

        with st.spinner(f"🔄 驗證 {get_exchange_display_name(exchange_type)} API 連線..."):
            adapter = get_adapter(exchange_type)
            # Bitget 需要額外的 password 參數
            if exchange_type == "bitget":
                adapter.init_exchange(api_key, api_secret, password=password)
            else:
                adapter.init_exchange(api_key, api_secret)
            adapter.load_markets()

            # 測試讀取餘額
            balances = adapter.fetch_balance()

            # 測試期貨權限
            try:
                positions = adapter.fetch_positions()
                futures_ok = True
            except Exception:
                futures_ok = False

        # 顯示驗證結果
        st.success(f"✅ {get_exchange_display_name(exchange_type)} API 驗證成功!")

        # 顯示餘額摘要
        total_balance = 0
        balance_info = []
        for currency in ["USDC", "USDT", "BTC", "ETH"]:
            if currency in balances:
                bal = balances[currency]
                if bal.wallet_balance > 0:
                    balance_info.append(f"{currency}: {bal.wallet_balance:.4f}")
                    if currency in ["USDC", "USDT"]:
                        total_balance += bal.wallet_balance

        if balance_info:
            st.info(f"💰 餘額: {' | '.join(balance_info[:3])}")

        if futures_ok:
            st.success("✅ 期貨交易權限正常")
        else:
            st.warning("⚠️ 無期貨交易權限，請確認 API 設定")

        return True

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ API 驗證失敗: {error_msg}")

        # 提供常見錯誤的解決建議
        if "Invalid API" in error_msg or "invalid" in error_msg.lower():
            st.warning("💡 建議: 請檢查 API Key 和 Secret 是否正確")
        elif "permission" in error_msg.lower() or "403" in error_msg:
            st.warning("💡 建議: 請確認 API 有期貨交易權限")
        elif "IP" in error_msg:
            st.warning("💡 建議: 請確認當前 IP 在 API 白名單中")
        elif "timestamp" in error_msg.lower() or "time" in error_msg.lower():
            st.warning("💡 建議: 請確認系統時間是否正確")

        return False


def test_api_connection(api_key: str, api_secret: str, exchange_type: str = "binance", password: str = ""):
    """測試 API 連線 (使用 Adapter)"""
    try:
        from exchanges import get_adapter, get_exchange_display_name

        with st.spinner(f"連接 {get_exchange_display_name(exchange_type)}..."):
            adapter = get_adapter(exchange_type)
            # Bitget 需要額外的 password 參數
            if exchange_type == "bitget":
                adapter.init_exchange(api_key, api_secret, password=password)
            else:
                adapter.init_exchange(api_key, api_secret)
            adapter.load_markets()

            balances = adapter.fetch_balance()

        st.success(f"✅ {get_exchange_display_name(exchange_type)} 連線成功!")

        # 顯示餘額
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, currency in enumerate(["USDC", "USDT", "BNB"]):
            if currency in balances:
                bal = balances[currency]
                if bal.wallet_balance > 0:
                    with cols[i % 3]:
                        st.metric(
                            currency,
                            f"{bal.wallet_balance:.4f}",
                            delta=f"可用: {bal.available_balance:.4f}"
                        )

        # 測試期貨權限
        try:
            positions = adapter.fetch_positions()
            st.success("✅ 期貨交易權限正常")
        except Exception:
            st.warning("⚠️ 無法讀取期貨倉位")

    except Exception as e:
        st.error(f"❌ 連線失敗: {str(e)}")


def render_max_enhancement():
    """渲染 MAX 增強功能設定"""
    st.subheader("⚡ MAX 增強功能")

    config = get_config()
    max_cfg = config.max_enhancement

    # 模式切換
    mode = st.toggle(
        "啟用增強模式",
        value=max_cfg.all_enhancements_enabled,
        help="開啟後啟用進階交易功能"
    )

    if mode != max_cfg.all_enhancements_enabled:
        max_cfg.all_enhancements_enabled = mode
        save_config()
        st.rerun()

    if max_cfg.all_enhancements_enabled:
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            funding = st.checkbox(
                "Funding Rate 偏向",
                value=max_cfg.funding_rate_enabled,
                help="根據資金費率調整開倉方向偏好"
            )
            if funding != max_cfg.funding_rate_enabled:
                max_cfg.funding_rate_enabled = funding
                save_config()

        with col2:
            glft = st.checkbox(
                "GLFT 庫存控制",
                value=max_cfg.glft_enabled,
                help="Gamma 調整庫存平衡機制"
            )
            if glft != max_cfg.glft_enabled:
                max_cfg.glft_enabled = glft
                save_config()

        with col3:
            dgt = st.checkbox(
                "動態網格",
                value=max_cfg.dynamic_grid_enabled,
                help="根據 ATR 自動調整網格間距"
            )
            if dgt != max_cfg.dynamic_grid_enabled:
                max_cfg.dynamic_grid_enabled = dgt
                save_config()

        # Gamma 參數
        if max_cfg.glft_enabled:
            gamma = st.slider(
                "Gamma (風險厭惡係數)",
                min_value=0.01,
                max_value=0.2,
                value=max_cfg.gamma,
                step=0.01,
                help="越大越傾向平衡多空倉位"
            )
            if gamma != max_cfg.gamma:
                max_cfg.gamma = gamma
                save_config()


def render_bandit_settings():
    """渲染 Bandit 學習設定"""
    st.subheader("🧠 Bandit 參數學習")

    config = get_config()
    bandit = config.bandit

    enabled = st.toggle(
        "啟用 UCB Bandit",
        value=bandit.enabled,
        help="自動學習最佳參數組合"
    )

    if enabled != bandit.enabled:
        bandit.enabled = enabled
        save_config()
        st.rerun()

    if bandit.enabled:
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            exploration = st.number_input(
                "探索係數",
                min_value=0.1,
                max_value=5.0,
                value=bandit.exploration_factor,
                step=0.1,
                help="越大越愛探索新參數"
            )
            if exploration != bandit.exploration_factor:
                bandit.exploration_factor = exploration
                save_config()

            window = st.number_input(
                "滑動窗口",
                min_value=10,
                max_value=200,
                value=bandit.window_size,
                step=10,
                help="考慮最近多少次交易"
            )
            if window != bandit.window_size:
                bandit.window_size = window
                save_config()

        with col2:
            contextual = st.checkbox(
                "Contextual (市場狀態感知)",
                value=bandit.contextual_enabled,
                help="根據市場狀態選擇不同策略"
            )
            if contextual != bandit.contextual_enabled:
                bandit.contextual_enabled = contextual
                save_config()

            thompson = st.checkbox(
                "Thompson Sampling",
                value=bandit.thompson_enabled,
                help="使用貝葉斯方法持續探索"
            )
            if thompson != bandit.thompson_enabled:
                bandit.thompson_enabled = thompson
                save_config()


def render_leading_indicator_settings():
    """渲染領先指標設定"""
    st.subheader("📡 領先指標")

    config = get_config()
    leading = config.leading_indicator

    enabled = st.toggle(
        "啟用領先指標",
        value=leading.enabled,
        help="OFI、成交量、價差分析"
    )

    if enabled != leading.enabled:
        leading.enabled = enabled
        save_config()
        st.rerun()

    if leading.enabled:
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**OFI 訂單流**")
            ofi_threshold = st.slider(
                "OFI 閾值",
                min_value=0.1,
                max_value=1.0,
                value=leading.ofi_threshold,
                step=0.1,
            )
            if ofi_threshold != leading.ofi_threshold:
                leading.ofi_threshold = ofi_threshold
                save_config()

        with col2:
            st.markdown("**成交量突增**")
            vol_threshold = st.slider(
                "成交量閾值 (倍)",
                min_value=1.0,
                max_value=5.0,
                value=leading.volume_surge_threshold,
                step=0.5,
            )
            if vol_threshold != leading.volume_surge_threshold:
                leading.volume_surge_threshold = vol_threshold
                save_config()

        with col3:
            st.markdown("**價差擴大**")
            spread_threshold = st.slider(
                "價差閾值 (倍)",
                min_value=1.0,
                max_value=3.0,
                value=leading.spread_surge_threshold,
                step=0.25,
            )
            if spread_threshold != leading.spread_surge_threshold:
                leading.spread_surge_threshold = spread_threshold
                save_config()


def render_risk_settings():
    """渲染風控設定"""
    st.subheader("🛡️ 風控設定")

    config = get_config()
    risk = config.risk

    enabled = st.toggle(
        "啟用追蹤止盈",
        value=risk.enabled,
        help="浮盈達標後自動追蹤止盈"
    )

    if enabled != risk.enabled:
        risk.enabled = enabled
        save_config()
        st.rerun()

    if risk.enabled:
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            margin = st.slider(
                "保證金閾值 (%)",
                min_value=10,
                max_value=80,
                value=int(risk.margin_threshold * 100),
                step=5,
                help="低於此比例時停止開新倉"
            )
            new_margin = margin / 100
            if new_margin != risk.margin_threshold:
                risk.margin_threshold = new_margin
                save_config()

        with col2:
            start_profit = st.number_input(
                "追蹤啟動 (U)",
                min_value=1.0,
                max_value=100.0,
                value=risk.trailing_start_profit,
                step=1.0,
                help="浮盈達到此值後開始追蹤"
            )
            if start_profit != risk.trailing_start_profit:
                risk.trailing_start_profit = start_profit
                save_config()

        with col3:
            drawdown = st.slider(
                "回撤觸發 (%)",
                min_value=1,
                max_value=20,
                value=int(risk.trailing_drawdown_pct * 100),
                step=1,
                help="從高點回撤此比例時止盈"
            )
            new_drawdown = drawdown / 100
            if new_drawdown != risk.trailing_drawdown_pct:
                risk.trailing_drawdown_pct = new_drawdown
                save_config()

        # === 硬止損設定 ===
        st.markdown("##### 🚨 硬止損")
        
        col1, col2 = st.columns(2)
        with col1:
            hard_stop = st.checkbox(
                "啟用硬止損",
                value=risk.hard_stop_enabled,
                help="當單方向浮虧超過閾值時強制平倉"
            )
            if hard_stop != risk.hard_stop_enabled:
                risk.hard_stop_enabled = hard_stop
                save_config()
        
        with col2:
            if risk.hard_stop_enabled:
                max_loss = st.slider(
                    "最大虧損 (%)",
                    min_value=1,
                    max_value=10,
                    value=int(risk.max_loss_pct * 100),
                    step=1,
                    help="單邊虧損達此比例時強制平倉"
                )
                new_max_loss = max_loss / 100
                if new_max_loss != risk.max_loss_pct:
                    risk.max_loss_pct = new_max_loss
                    save_config()

        # === 趨勢過濾器 (實驗性) ===
        st.markdown("##### 🧪 趨勢過濾器 (實驗性)")
        st.caption("⚠️ 此功能可能改變網格策略的本質。啟用後會根據 MA 方向限制開倉方向。")
        
        # 趨勢過濾需要在 SymbolConfig 層級設置，這裡只顯示說明
        st.info("""
        **趨勢過濾器說明:**
        - 價格 > MA200: 只做多
        - 價格 < MA200: 只做空
        - 預設關閉，可在配置文件中啟用
        
        配置項: `trend_filter_enabled`, `trend_ma_period`, `trend_buffer_pct`
        """)


def main():
    """主函數"""
    # 先渲染側邊欄
    render_sidebar()

    st.title("🛠️ 設定")

    # === 推薦交易所區塊 (最上方) ===
    render_exchange_referrals()

    st.divider()

    # 標籤頁
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 API",
        "⚡ MAX 增強",
        "🧠 Bandit 學習",
        "📡 領先指標",
        "🛡️ 風控"
    ])

    with tab1:
        render_api_settings()

    with tab2:
        render_max_enhancement()

    with tab3:
        render_bandit_settings()

    with tab4:
        render_leading_indicator_settings()

    with tab5:
        render_risk_settings()

    # 底部操作
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 重新載入配置"):
            reload_config()
            st.success("配置已重新載入")
            st.rerun()

    with col2:
        if st.button("💾 強制保存"):
            save_config()
            st.success("配置已保存")


def get_logo_base64(logo_name: str) -> str:
    """讀取 Logo 並轉為 base64"""
    import base64
    logo_path = Path(__file__).parent.parent / "assets" / "logos" / logo_name
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def render_exchange_referrals():
    """渲染交易所推薦連結"""
    st.subheader("🏦 支援的交易所")
    st.caption("使用推薦連結註冊可獲得手續費優惠")

    # 交易所資訊
    exchanges = [
        {
            "name": "Binance",
            "logo": "Binance.png",
            "referral": "https://accounts.binance.com/register?ref=ASLOUIS",
            "status": "✅ 已支援",
        },
        {
            "name": "Bybit",
            "logo": "bybit.png",
            "referral": "https://www.bybit.com/invite?ref=B1MDMYE",
            "status": "✅ 已支援",
        },
        {
            "name": "Bitget",
            "logo": "bitget.png",
            "referral": "https://partner.bitget.fit/bg/aslouis",
            "status": "✅ 已支援",
        },
        {
            "name": "Gate.io",
            "logo": "gate.png",
            "referral": "https://www.gatenode.xyz/signup/VLUSXFLFAQ?ref_type=103",
            "status": "✅ 已支援",
        },
    ]

    # 顯示 4 個交易所卡片
    cols = st.columns(4)

    for i, ex in enumerate(exchanges):
        with cols[i]:
            status_color = '#00D68F' if '已支援' in ex['status'] else '#8B8D97'
            logo_b64 = get_logo_base64(ex['logo'])

            st.markdown(f"""
            <div style="
                background: linear-gradient(145deg, #1E2229 0%, #171A1F 100%);
                border-radius: 12px;
                padding: 20px 16px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.05);
                min-height: 200px;
            ">
                <img src="data:image/png;base64,{logo_b64}" style="
                    width: 56px;
                    height: 56px;
                    border-radius: 12px;
                    margin-bottom: 12px;
                    object-fit: contain;
                ">
                <div style="
                    font-size: 16px;
                    font-weight: 600;
                    color: #FFFFFF;
                    margin-bottom: 4px;
                ">{ex['name']}</div>
                <div style="
                    font-size: 12px;
                    color: {status_color};
                    margin-bottom: 16px;
                ">{ex['status']}</div>
                <a href="{ex['referral']}" target="_blank" style="
                    display: inline-block;
                    background: linear-gradient(135deg, #6C63FF 0%, #5B54E8 100%);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-size: 13px;
                    font-weight: 600;
                ">註冊領優惠</a>
            </div>
            """, unsafe_allow_html=True)


# 執行頁面
main()
