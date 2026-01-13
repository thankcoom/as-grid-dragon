# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
工具函數
========
通用工具函數
"""

from typing import Tuple, Optional, Union
from constants import SYMBOL_MAP


def safe_float(value: Union[str, int, float, None], default: float = 0.0) -> float:
    """
    安全地將值轉換為 float，處理 None 和空字串

    Args:
        value: 要轉換的值 (可能是 None, "", int, float, str)
        default: 轉換失敗時的預設值

    Returns:
        轉換後的 float 值，失敗時返回 default

    Examples:
        >>> safe_float(None)
        0.0
        >>> safe_float("")
        0.0
        >>> safe_float("3.14")
        3.14
        >>> safe_float({"key": "value"}.get("missing"), 1.0)
        1.0
    """
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_symbol(symbol_input: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    標準化交易對符號
    
    Args:
        symbol_input: 用戶輸入的交易對符號 (如 "XRPUSDC", "XRP/USDC", "xrp-usdc")
    
    Returns:
        (raw_symbol, ccxt_symbol, coin, quote)
        - raw_symbol: 標準化後的符號 (如 "XRPUSDC")
        - ccxt_symbol: CCXT 格式 (如 "XRP/USDC:USDC")
        - coin: 幣種 (如 "XRP")
        - quote: 報價幣種 (如 "USDC")
        
        如果無法識別，返回 (None, None, None, None)
    """
    s = symbol_input.upper().strip().replace("/", "").replace(":", "").replace("-", "")

    if s in SYMBOL_MAP:
        ccxt_sym = SYMBOL_MAP[s]
        parts = ccxt_sym.split("/")
        coin = parts[0]
        quote = parts[1].split(":")[0]
        return s, ccxt_sym, coin, quote

    for suffix in ["USDC", "USDT"]:
        if s.endswith(suffix):
            coin = s[:-len(suffix)]
            if coin:
                ccxt_sym = f"{coin}/{suffix}:{suffix}"
                return s, ccxt_sym, coin, suffix

    return None, None, None, None
