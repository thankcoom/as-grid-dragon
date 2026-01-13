"""核心交易模組"""
# 注意：使用延遲導入避免循環依賴
# 直接引用時使用: from AS刷怪籠.core.strategy import GridStrategy

__all__ = ["GridStrategy", "MaxGridBot", "BacktestManager"]

def __getattr__(name):
    if name == "GridStrategy":
        from .strategy import GridStrategy
        return GridStrategy
    elif name == "MaxGridBot":
        from .bot import MaxGridBot
        return MaxGridBot
    elif name == "BacktestManager":
        from .backtest import BacktestManager
        return BacktestManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
