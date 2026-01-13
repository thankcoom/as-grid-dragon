"""
UI 主題模組
===========
自訂 CSS 樣式，打造專業交易儀表板風格
"""

import streamlit as st


def apply_custom_theme():
    """套用自訂主題樣式"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 自訂 CSS 樣式
CUSTOM_CSS = """
<style>
/* ========== 全局樣式 ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.main {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 隱藏 Streamlit 預設元素 (保留 header 的展開按鈕) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 只隱藏 header 的裝飾元素，保留功能按鈕 */
header [data-testid="stDecoration"] {
    display: none;
}

/* 首頁導航項目 - 將 "app" 改名為 "🏠 控制中心" */
[data-testid="stSidebarNav"] li:first-child a span,
[data-testid="stSidebarNav"] li:first-child a div {
    font-size: 0 !important;
}

[data-testid="stSidebarNav"] li:first-child a span::before,
[data-testid="stSidebarNav"] li:first-child a div::before {
    content: "🏠 控制中心";
    font-size: 14px !important;
    color: #AAAAAA !important;
}

/* 首頁導航項目選中時 - 白色文字 */
[data-testid="stSidebarNav"] li:first-child a[aria-selected="true"] span::before,
[data-testid="stSidebarNav"] li:first-child a[aria-selected="true"] div::before,
[data-testid="stSidebarNav"] li:first-child a[aria-current="page"] span::before,
[data-testid="stSidebarNav"] li:first-child a[aria-current="page"] div::before {
    color: #FFFFFF !important;
}

/* ========== 卡片樣式 ========== */
.card {
    background: linear-gradient(145deg, #1E2229 0%, #171A1F 100%);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

.card-header {
    font-size: 14px;
    font-weight: 500;
    color: #8B8D97;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.card-value {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
}

.card-positive {
    color: #00D68F !important;
}

.card-negative {
    color: #FF4D4F !important;
}

/* ========== 狀態指示器 ========== */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}

.status-running {
    background: rgba(0, 214, 143, 0.15);
    color: #00D68F;
    border: 1px solid rgba(0, 214, 143, 0.3);
}

.status-stopped {
    background: rgba(139, 141, 151, 0.15);
    color: #8B8D97;
    border: 1px solid rgba(139, 141, 151, 0.3);
}

.status-warning {
    background: rgba(255, 193, 7, 0.15);
    color: #FFC107;
    border: 1px solid rgba(255, 193, 7, 0.3);
}

/* ========== 按鈕樣式 ========== */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #5B54E8 100%);
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
}

.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: none;
}

/* ========== Metric 樣式優化 ========== */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1E2229 0%, #171A1F 100%);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    color: #8B8D97 !important;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] > div {
    font-size: 14px !important;
    font-weight: 600;
}

/* ========== 表格樣式 ========== */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

.stDataFrame [data-testid="stTable"] {
    background: #1A1D24;
}

/* ========== Expander 樣式 ========== */
.streamlit-expanderHeader {
    background: linear-gradient(145deg, #1E2229 0%, #1A1D24 100%);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    font-weight: 600;
}

/* ========== Tab 樣式 ========== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 12px 24px;
    border: 1px solid rgba(255,255,255,0.05);
    color: #8B8D97 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #FAFAFA !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF 0%, #5B54E8 100%);
    border-color: transparent;
    color: #FFFFFF !important;
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
    color: #FFFFFF !important;
}

/* ========== 輸入框樣式 ========== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #1A1D24;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: #FAFAFA;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #6C63FF;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2);
}

/* ========== 進度條 ========== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6C63FF 0%, #00D68F 100%);
    border-radius: 10px;
}

/* ========== 側邊欄 ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12151A 0%, #0E1117 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px;
}

/* 側邊欄導航項目樣式 */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p,
[data-testid="stSidebarNav"] li a {
    color: #AAAAAA !important;
    transition: all 0.2s ease;
}

[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a:hover span,
[data-testid="stSidebarNav"] a:hover p {
    color: #FFFFFF !important;
    background: rgba(108, 99, 255, 0.15) !important;
}

/* 選中的導航項目 - 白色文字 (加強選擇器) */
[data-testid="stSidebarNav"] a[aria-selected="true"],
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNav"] li[data-testid="stSidebarNavLink"] a[aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.3) 0%, rgba(91, 84, 232, 0.3) 100%) !important;
    border-left: 3px solid #6C63FF;
}

[data-testid="stSidebarNav"] a[aria-selected="true"] *,
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNav"] a[aria-selected="true"] span,
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNav"] a[aria-selected="true"] p,
[data-testid="stSidebarNav"] a[aria-current="page"] p {
    color: #FFFFFF !important;
}

/* 確保所有導航文字可見 */
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNav"] p {
    color: inherit !important;
}

/* ========== 分隔線 ========== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 24px 0;
}

/* ========== 成功/錯誤/警告訊息 ========== */
.stSuccess {
    background: rgba(0, 214, 143, 0.1);
    border: 1px solid rgba(0, 214, 143, 0.2);
    border-radius: 10px;
}

.stError {
    background: rgba(255, 77, 79, 0.1);
    border: 1px solid rgba(255, 77, 79, 0.2);
    border-radius: 10px;
}

.stWarning {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid rgba(255, 193, 7, 0.2);
    border-radius: 10px;
}

.stInfo {
    background: rgba(108, 99, 255, 0.1);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 10px;
}

/* ========== 標題樣式 ========== */
h1 {
    font-weight: 700 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #A8A8A8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2, h3 {
    font-weight: 600 !important;
    color: #FAFAFA !important;
}

/* ========== 動畫效果 ========== */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.slide-in {
    animation: slideIn 0.3s ease-out;
}
</style>
"""


def render_status_badge(status: str, text: str) -> str:
    """渲染狀態徽章"""
    status_class = {
        "running": "status-running",
        "stopped": "status-stopped",
        "warning": "status-warning",
    }.get(status, "status-stopped")

    return f'<span class="status-badge {status_class}">{text}</span>'


def render_metric_card(label: str, value: str, delta: str = None, positive: bool = True) -> str:
    """渲染指標卡片"""
    delta_html = ""
    if delta:
        delta_class = "card-positive" if positive else "card-negative"
        delta_html = f'<div class="{delta_class}" style="font-size: 14px; margin-top: 4px;">{delta}</div>'

    return f"""
    <div class="card">
        <div class="card-header">{label}</div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>
    """


def render_header_with_logo(title: str, subtitle: str = None):
    """渲染帶 Logo 的標題"""
    subtitle_html = f'<p style="color: #8B8D97; margin-top: 4px;">{subtitle}</p>' if subtitle else ""

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px;">
        <div style="
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #6C63FF 0%, #5B54E8 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        ">📊</div>
        <div>
            <h1 style="margin: 0; font-size: 28px;">{title}</h1>
            {subtitle_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
