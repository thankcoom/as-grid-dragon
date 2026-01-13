"""指標與優化模組"""
# 使用延遲導入避免循環依賴和過早載入外部套件

__all__ = [
    "UCBBanditOptimizer",
    "ParameterArm", 
    "MarketContext",
    "LeadingIndicatorManager",
    "FundingRateManager",
    "GLFTController",
    "DGTBoundaryManager",
    "DynamicGridManager",
]

def __getattr__(name):
    if name in ("UCBBanditOptimizer", "ParameterArm", "MarketContext"):
        from . import bandit
        return getattr(bandit, name)
    elif name == "LeadingIndicatorManager":
        from .leading import LeadingIndicatorManager
        return LeadingIndicatorManager
    elif name in ("FundingRateManager", "GLFTController"):
        from . import funding
        return getattr(funding, name)
    elif name in ("DGTBoundaryManager", "DynamicGridManager"):
        from . import dgt
        return getattr(dgt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
