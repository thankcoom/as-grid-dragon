"""
共用側邊欄組件
==============
所有頁面統一使用的側邊欄
"""

import streamlit as st


def render_sidebar():
    """渲染共用側邊欄"""
    with st.sidebar:
        # Threads 連結在最上方
        st.markdown("""
        <a href="https://www.threads.com/@mr.__.l" target="_blank" style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 16px;
            background: linear-gradient(135deg, rgba(108,99,255,0.2) 0%, rgba(91,84,232,0.2) 100%);
            border: 1px solid rgba(108,99,255,0.3);
            border-radius: 10px;
            color: #FAFAFA;
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 16px;
            transition: all 0.2s;
        ">
            <span>🧵</span> 追蹤 @mr.__.l
        </a>
        """, unsafe_allow_html=True)

        # Logo 和品牌
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="
                width: 50px;
                height: 50px;
                background: linear-gradient(135deg, #6C63FF 0%, #5B54E8 100%);
                border-radius: 14px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                margin-bottom: 10px;
            ">📊</div>
            <h3 style="margin: 0; font-size: 18px;">Louis Grid</h3>
            <p style="color: #8B8D97; font-size: 11px; margin-top: 4px;">MAX 增強版</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 系統狀態 (需要從 state 導入)
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from state import get_config, is_trading_active

            config = get_config()
            max_cfg = config.max_enhancement

            st.markdown("**系統狀態**")

            if is_trading_active():
                st.markdown("""
                <span style="
                    display: inline-flex;
                    align-items: center;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 13px;
                    font-weight: 600;
                    background: rgba(0, 214, 143, 0.15);
                    color: #00D68F;
                    border: 1px solid rgba(0, 214, 143, 0.3);
                ">交易運行中</span>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <span style="
                    display: inline-flex;
                    align-items: center;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 13px;
                    font-weight: 600;
                    background: rgba(139, 141, 151, 0.15);
                    color: #8B8D97;
                    border: 1px solid rgba(139, 141, 151, 0.3);
                ">待命中</span>
                """, unsafe_allow_html=True)

            st.divider()

            # 啟用的功能
            st.markdown("**啟用功能**")

            features = []
            if max_cfg.all_enhancements_enabled:
                features.append("✓ 增強模式")
                if max_cfg.funding_rate_enabled:
                    features.append("  • Funding Rate")
                if max_cfg.glft_enabled:
                    features.append("  • GLFT 庫存控制")
                if max_cfg.dynamic_grid_enabled:
                    features.append("  • 動態網格")
            else:
                features.append("○ 純淨模式")

            if config.bandit.enabled:
                features.append("✓ Bandit 學習")

            if config.leading_indicator.enabled:
                features.append("✓ 領先指標")

            for f in features:
                st.caption(f)

            st.divider()

        except Exception:
            pass

        # 版本資訊
        st.caption("v1.0.0 | Web UI")
