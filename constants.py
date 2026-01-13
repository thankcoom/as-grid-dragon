# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
常量定義
========
交易對映射、路徑配置等全局常量
"""

from pathlib import Path
import os
import logging

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              交易對映射                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 支援的交易對 (簡化格式 -> ccxt格式)
SYMBOL_MAP = {
    "XRPUSDC": "XRP/USDC:USDC",
    "BTCUSDC": "BTC/USDC:USDC",
    "ETHUSDC": "ETH/USDC:USDC",
    "SOLUSDC": "SOL/USDC:USDC",
    "DOGEUSDC": "DOGE/USDC:USDC",
    "XRPUSDT": "XRP/USDT:USDT",
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
    "DOGEUSDT": "DOGE/USDT:USDT",
    "BNBUSDT": "BNB/USDT:USDT",
    "ADAUSDT": "ADA/USDT:USDT",
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              路徑配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 配置文件路徑
CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "trading_config_max.json"
DATA_DIR = Path(__file__).parent.parent / "asBack" / "data"

# 創建目錄
CONFIG_DIR.mkdir(exist_ok=True)
os.makedirs(Path(__file__).parent.parent / "log", exist_ok=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              日誌配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

LOG_DIR = Path(__file__).parent.parent / "log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.FileHandler(LOG_DIR / "as_grid_max.log")]
)

logger = logging.getLogger("as_grid_max")
