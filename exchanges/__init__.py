# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Exchange Adapters Package
=========================
多交易所適配器工廠

使用方式:
    from exchanges import get_adapter, list_supported_exchanges

    # 取得支援的交易所列表
    exchanges = list_supported_exchanges()
    # ['binance', 'bybit', 'okx', ...]

    # 取得特定交易所的適配器
    adapter = get_adapter("binance")
    adapter.init_exchange(api_key, api_secret)
"""

from typing import Dict, Type, List
from .base import ExchangeAdapter

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              交易所註冊表                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 延遲載入的交易所映射
# key: 交易所名稱 (小寫)
# value: (模組路徑, 類別名稱)
_ADAPTER_REGISTRY: Dict[str, tuple] = {
    "binance": ("exchanges.binance", "BinanceAdapter"),
    "bybit": ("exchanges.bybit", "BybitAdapter"),
    "bitget": ("exchanges.bitget", "BitgetAdapter"),
    "gate": ("exchanges.gate", "GateAdapter"),
}

# 交易所顯示名稱
_DISPLAY_NAMES: Dict[str, str] = {
    "binance": "Binance",
    "bybit": "Bybit",
    "bitget": "Bitget",
    "gate": "Gate.io",
}

# 推薦連結 (用於 UI 顯示)
_REFERRAL_LINKS: Dict[str, str] = {
    "binance": "https://accounts.binance.com/register?ref=ASLOUIS",
    "bybit": "https://www.bybit.com/invite?ref=B1MDMYE",
    "bitget": "https://partner.bitget.fit/bg/aslouis",
    "gate": "https://www.gatenode.xyz/signup/VLUSXFLFAQ?ref_type=103",
}

# 已載入的適配器類別快取
_loaded_adapters: Dict[str, Type[ExchangeAdapter]] = {}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              公開 API                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_adapter(exchange_type: str) -> ExchangeAdapter:
    """
    取得交易所適配器實例

    Args:
        exchange_type: 交易所類型 (e.g., "binance", "bybit")

    Returns:
        ExchangeAdapter: 交易所適配器實例

    Raises:
        ValueError: 不支援的交易所類型
        ImportError: 適配器模組載入失敗
    """
    exchange_type = exchange_type.lower()

    if exchange_type not in _ADAPTER_REGISTRY:
        supported = ", ".join(_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"不支援的交易所: {exchange_type}。"
            f"支援的交易所: {supported}"
        )

    # 延遲載入適配器類別
    if exchange_type not in _loaded_adapters:
        module_path, class_name = _ADAPTER_REGISTRY[exchange_type]
        _loaded_adapters[exchange_type] = _load_adapter_class(module_path, class_name)

    # 返回新實例
    adapter_class = _loaded_adapters[exchange_type]
    return adapter_class()


def list_supported_exchanges() -> List[str]:
    """
    列出所有支援的交易所

    Returns:
        List[str]: 交易所名稱列表
    """
    return list(_ADAPTER_REGISTRY.keys())


def get_exchange_display_name(exchange_type: str) -> str:
    """
    取得交易所的顯示名稱

    Args:
        exchange_type: 交易所類型

    Returns:
        str: 顯示名稱 (e.g., "Binance")
    """
    return _DISPLAY_NAMES.get(exchange_type.lower(), exchange_type.title())


def is_exchange_supported(exchange_type: str) -> bool:
    """
    檢查交易所是否支援

    Args:
        exchange_type: 交易所類型

    Returns:
        bool: 是否支援
    """
    return exchange_type.lower() in _ADAPTER_REGISTRY


def get_referral_link(exchange_type: str) -> str:
    """
    取得交易所的推薦連結

    Args:
        exchange_type: 交易所類型

    Returns:
        str: 推薦連結 URL
    """
    return _REFERRAL_LINKS.get(exchange_type.lower(), "")


def list_all_exchanges() -> List[str]:
    """
    列出所有交易所 (包含未實作的)

    Returns:
        List[str]: 所有交易所名稱
    """
    return list(_DISPLAY_NAMES.keys())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              內部函數                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _load_adapter_class(module_path: str, class_name: str) -> Type[ExchangeAdapter]:
    """
    動態載入適配器類別

    Args:
        module_path: 模組路徑 (e.g., "exchanges.binance")
        class_name: 類別名稱 (e.g., "BinanceAdapter")

    Returns:
        Type[ExchangeAdapter]: 適配器類別

    Raises:
        ImportError: 載入失敗
    """
    import importlib

    try:
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)

        # 驗證是否為 ExchangeAdapter 子類
        if not issubclass(adapter_class, ExchangeAdapter):
            raise TypeError(
                f"{class_name} 必須繼承 ExchangeAdapter"
            )

        return adapter_class

    except ImportError as e:
        raise ImportError(
            f"無法載入交易所適配器 {module_path}.{class_name}: {e}"
        ) from e
    except AttributeError as e:
        raise ImportError(
            f"模組 {module_path} 中找不到類別 {class_name}: {e}"
        ) from e


# 導出公開 API
__all__ = [
    "ExchangeAdapter",
    "get_adapter",
    "list_supported_exchanges",
    "list_all_exchanges",
    "get_exchange_display_name",
    "get_referral_link",
    "is_exchange_supported",
]
