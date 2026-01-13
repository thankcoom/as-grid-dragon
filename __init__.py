"""
AS 刷怪籠 - 網格交易系統 MAX 版本
====================================
模組化重構版本，基於 as_terminal_max.py

功能:
1. 雙向網格策略 (多空同時運行)
2. UCB Bandit 參數自動優化
3. 領先指標系統 (OFI, Volume, Spread)
4. GLFT 庫存控制
5. 動態網格邊界管理

使用:
    python -m AS刷怪籠.main
"""

__version__ = "2.0.0"
__author__ = "AS Grid Trading"

# 延遲載入避免循環依賴
__all__ = [
    "GridStrategy",
    "MaxGridBot", 
    "GlobalConfig",
    "SymbolConfig",
]

def __getattr__(name):
    if name == "GridStrategy":
        from .core.strategy import GridStrategy
        return GridStrategy
    elif name == "MaxGridBot":
        from .core.bot import MaxGridBot
        return MaxGridBot
    elif name == "GlobalConfig":
        from .config.models import GlobalConfig
        return GlobalConfig
    elif name == "SymbolConfig":
        from .config.models import SymbolConfig
        return SymbolConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
