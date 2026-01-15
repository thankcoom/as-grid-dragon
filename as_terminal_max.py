"""
AS 網格交易系統 - MAX 版本
==============================
基於 as_terminal_pro.py 增強版本

新增功能:
1. Funding Rate 偏向機制 - 根據資金費率調整多空偏好
2. GLFT 風險係數 γ - 更精細的庫存控制
3. 動態網格範圍 - ATR/波動率自適應

依賴:
-----
pip install rich ccxt websockets pandas numpy

使用:
-----
python as_terminal_max.py
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              標準庫導入                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import asyncio
import json
import logging
import time
import math
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              第三方庫導入                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import ccxt
import websockets
import ssl
import certifi
import pandas as pd
import numpy as np

# Rich 終端美化
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              常量定義                                          ║
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

# 配置文件路徑
CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_FILE = CONFIG_DIR / "trading_config_max.json"
DATA_DIR = Path(__file__).parent / "asBack" / "data"

# 創建目錄
CONFIG_DIR.mkdir(exist_ok=True)
os.makedirs("log", exist_ok=True)

# Console
console = Console()

# 日誌配置
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.FileHandler("log/as_terminal_max.log")]
)
logger = logging.getLogger("as_grid_max")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              工具函數                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def normalize_symbol(symbol_input: str) -> tuple:
    """標準化交易對符號"""
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         網格策略核心 (統一回測/實盤邏輯)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GridStrategy:
    """
    網格策略核心邏輯 - 統一回測與實盤

    此類提取所有策略計算邏輯，確保回測與實盤行為一致。
    不包含任何 I/O 操作（下單、日誌等），只負責純計算。

    使用方式:
    - 回測: BacktestManager 調用靜態方法計算價格/數量
    - 實盤: MaxGridBot 調用相同方法，確保邏輯一致
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 常量定義 - 集中管理魔術數字
    # ═══════════════════════════════════════════════════════════════════════════
    DEAD_MODE_FALLBACK_LONG = 1.05    # 多頭裝死模式無對手倉時的止盈比例
    DEAD_MODE_FALLBACK_SHORT = 0.95   # 空頭裝死模式無對手倉時的止盈比例
    DEAD_MODE_DIVISOR = 100           # 裝死模式計算除數 (持倉比/100)

    @staticmethod
    def is_dead_mode(position: float, threshold: float) -> bool:
        """
        判斷是否進入裝死模式

        Args:
            position: 當前持倉量
            threshold: 裝死閾值 (position_threshold)

        Returns:
            True = 進入裝死模式，停止補倉
        """
        return position > threshold

    @staticmethod
    def calculate_dead_mode_price(
        base_price: float,
        my_position: float,
        opposite_position: float,
        side: str
    ) -> float:
        """
        計算裝死模式的特殊止盈價格

        公式:
        - 多頭: price × ((多倉/空倉)/100 + 1)，無空倉時 price × 1.05
        - 空頭: price ÷ ((空倉/多倉)/100 + 1)，無多倉時 price × 0.95

        設計理念:
        - 持倉比例越失衡，止盈價格越遠
        - 等待極端反彈才平倉，避免在不利位置出場

        Args:
            base_price: 基準價格 (當前價或上次網格價)
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            side: 'long' 或 'short'

        Returns:
            特殊止盈價格
        """
        if opposite_position > 0:
            r = (my_position / opposite_position) / GridStrategy.DEAD_MODE_DIVISOR + 1
            if side == 'long':
                return base_price * r
            else:  # short
                return base_price / r
        else:
            # 無對手倉，使用固定比例
            if side == 'long':
                return base_price * GridStrategy.DEAD_MODE_FALLBACK_LONG
            else:  # short
                return base_price * GridStrategy.DEAD_MODE_FALLBACK_SHORT

    @staticmethod
    def calculate_tp_quantity(
        base_qty: float,
        my_position: float,
        opposite_position: float,
        position_limit: float,
        position_threshold: float
    ) -> float:
        """
        計算止盈數量

        加倍條件 (滿足任一):
        1. 本方向持倉 > position_limit
        2. 對手方向持倉 >= position_threshold

        設計理念:
        - 持倉過大時加速出場
        - 對手進入裝死時也加速，維持多空平衡

        Args:
            base_qty: 基礎數量 (initial_quantity)
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            position_limit: 持倉上限 (觸發加倍)
            position_threshold: 裝死閾值 (對手觸發時也加倍)

        Returns:
            調整後的止盈數量 (1x 或 2x)
        """
        if my_position > position_limit or opposite_position >= position_threshold:
            return base_qty * 2
        return base_qty

    @staticmethod
    def calculate_grid_prices(
        base_price: float,
        take_profit_spacing: float,
        grid_spacing: float,
        side: str
    ) -> Tuple[float, float]:
        """
        計算正常模式的網格價格

        Args:
            base_price: 基準價格
            take_profit_spacing: 止盈間距 (如 0.004 = 0.4%)
            grid_spacing: 補倉間距 (如 0.006 = 0.6%)
            side: 'long' 或 'short'

        Returns:
            (止盈價格, 補倉價格)
        """
        if side == 'long':
            tp_price = base_price * (1 + take_profit_spacing)
            entry_price = base_price * (1 - grid_spacing)
        else:  # short
            tp_price = base_price * (1 - take_profit_spacing)
            entry_price = base_price * (1 + grid_spacing)

        return tp_price, entry_price

    @staticmethod
    def get_grid_decision(
        price: float,
        my_position: float,
        opposite_position: float,
        position_threshold: float,
        position_limit: float,
        base_qty: float,
        take_profit_spacing: float,
        grid_spacing: float,
        side: str
    ) -> dict:
        """
        獲取完整的網格決策 (主要入口方法)

        統一回測與實盤的決策邏輯，返回所有需要的計算結果。
        此方法是無副作用的純函數，可用於回測和實盤。

        Args:
            price: 當前/基準價格
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            position_threshold: 裝死閾值
            position_limit: 持倉上限
            base_qty: 基礎數量
            take_profit_spacing: 止盈間距
            grid_spacing: 補倉間距
            side: 'long' 或 'short'

        Returns:
            {
                'dead_mode': bool,       # 是否裝死模式
                'tp_price': float,       # 止盈價格
                'entry_price': float,    # 補倉價格 (裝死模式為 None)
                'tp_qty': float,         # 止盈數量
                'entry_qty': float,      # 補倉數量 (裝死模式為 0)
            }

        Example:
            >>> decision = GridStrategy.get_grid_decision(
            ...     price=2.5, my_position=100, opposite_position=50,
            ...     position_threshold=500, position_limit=100,
            ...     base_qty=10, take_profit_spacing=0.004,
            ...     grid_spacing=0.006, side='long'
            ... )
            >>> decision['dead_mode']
            False
            >>> decision['tp_price']
            2.51  # 2.5 * 1.004
        """
        dead_mode = GridStrategy.is_dead_mode(my_position, position_threshold)

        tp_qty = GridStrategy.calculate_tp_quantity(
            base_qty, my_position, opposite_position,
            position_limit, position_threshold
        )

        if dead_mode:
            tp_price = GridStrategy.calculate_dead_mode_price(
                price, my_position, opposite_position, side
            )
            return {
                'dead_mode': True,
                'tp_price': tp_price,
                'entry_price': None,
                'tp_qty': tp_qty,
                'entry_qty': 0,
            }
        else:
            tp_price, entry_price = GridStrategy.calculate_grid_prices(
                price, take_profit_spacing, grid_spacing, side
            )
            return {
                'dead_mode': False,
                'tp_price': tp_price,
                'entry_price': entry_price,
                'tp_qty': tp_qty,
                'entry_qty': base_qty,
            }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAX 增強模組                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MaxEnhancement:
    """
    MAX 版本增強功能配置

    1. Funding Rate 偏向
    2. GLFT γ 風險係數
    3. 動態網格範圍 (已被領先指標取代)

    建議配置:
    - all_enhancements_enabled: False (保持無腦執行)
    - 使用 Bandit + 領先指標 即可
    """
    # === 主開關 ===
    all_enhancements_enabled: bool = False   # 總開關：False = 純淨模式 (保持無腦執行)

    # === Funding Rate 偏向 ===
    funding_rate_enabled: bool = False          # 預設關閉 (長期持倉時可開啟)
    funding_rate_threshold: float = 0.0001      # 0.01% 以上才調整
    funding_rate_position_bias: float = 0.2     # 偏向調整比例 (20%)

    # === GLFT γ 風險係數 ===
    glft_enabled: bool = False                  # 預設關閉 (多空不平衡時可開啟)
    gamma: float = 0.1                          # 風險厭惡係數 (0.01-1.0)
    inventory_target: float = 0.5               # 目標庫存比例 (0.5 = 多空平衡)

    # === 動態網格範圍 (ATR - 滯後指標) ===
    dynamic_grid_enabled: bool = False          # 預設關閉 (已被領先指標取代)
    atr_period: int = 14                        # ATR 週期
    atr_multiplier: float = 1.5                 # ATR 乘數
    min_spacing: float = 0.002                  # 最小間距 0.2%
    max_spacing: float = 0.015                  # 最大間距 1.5%
    volatility_lookback: int = 100              # 波動率回看期

    def to_dict(self) -> dict:
        return {
            "all_enhancements_enabled": self.all_enhancements_enabled,
            "funding_rate_enabled": self.funding_rate_enabled,
            "funding_rate_threshold": self.funding_rate_threshold,
            "funding_rate_position_bias": self.funding_rate_position_bias,
            "glft_enabled": self.glft_enabled,
            "gamma": self.gamma,
            "inventory_target": self.inventory_target,
            "dynamic_grid_enabled": self.dynamic_grid_enabled,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "min_spacing": self.min_spacing,
            "max_spacing": self.max_spacing,
            "volatility_lookback": self.volatility_lookback
        }

    def is_feature_enabled(self, feature: str) -> bool:
        """檢查功能是否啟用 (考慮總開關)"""
        if not self.all_enhancements_enabled:
            return False
        return getattr(self, f"{feature}_enabled", False)

    @classmethod
    def from_dict(cls, data: dict) -> 'MaxEnhancement':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         UCB Bandit 參數優化器                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class BanditConfig:
    """
    Bandit 優化器配置 (增強版)

    新增功能:
    1. 冷啟動預載 - 首次運行使用歷史最佳參數
    2. Contextual - 根據市場狀態選擇不同策略
    3. Thompson Sampling - 連續參數空間探索
    4. MDD 懲罰 - 改進 reward 計算
    """
    enabled: bool = True
    window_size: int = 50              # 滑動窗口大小 (只看最近 N 筆交易)
    exploration_factor: float = 1.5    # UCB 探索係數 (越大越愛探索)
    min_pulls_per_arm: int = 3         # 每個 arm 至少要試幾次
    update_interval: int = 10          # 每 N 筆交易評估一次

    # === 冷啟動配置 ===
    cold_start_enabled: bool = True    # 啟用冷啟動預載
    cold_start_arm_idx: int = 4        # 預設使用的 arm 索引 (平衡型)

    # === Contextual Bandit ===
    contextual_enabled: bool = True    # 啟用市場狀態感知
    volatility_lookback: int = 20      # 波動率計算回看期
    trend_lookback: int = 50           # 趨勢計算回看期
    high_volatility_threshold: float = 0.02  # 高波動閾值 (2%)
    trend_threshold: float = 0.01      # 趨勢閾值 (1%)

    # === Thompson Sampling ===
    thompson_enabled: bool = True      # 啟用 Thompson Sampling
    thompson_prior_alpha: float = 1.0  # Beta 分布先驗 α
    thompson_prior_beta: float = 1.0   # Beta 分布先驗 β
    param_perturbation: float = 0.1    # 參數擾動範圍 (10%)

    # === Reward 改進 ===
    mdd_penalty_weight: float = 0.5    # Max Drawdown 懲罰權重
    win_rate_bonus: float = 0.2        # 勝率獎勵權重

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "exploration_factor": self.exploration_factor,
            "min_pulls_per_arm": self.min_pulls_per_arm,
            "update_interval": self.update_interval,
            "cold_start_enabled": self.cold_start_enabled,
            "cold_start_arm_idx": self.cold_start_arm_idx,
            "contextual_enabled": self.contextual_enabled,
            "volatility_lookback": self.volatility_lookback,
            "trend_lookback": self.trend_lookback,
            "high_volatility_threshold": self.high_volatility_threshold,
            "trend_threshold": self.trend_threshold,
            "thompson_enabled": self.thompson_enabled,
            "thompson_prior_alpha": self.thompson_prior_alpha,
            "thompson_prior_beta": self.thompson_prior_beta,
            "param_perturbation": self.param_perturbation,
            "mdd_penalty_weight": self.mdd_penalty_weight,
            "win_rate_bonus": self.win_rate_bonus
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BanditConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# 市場狀態枚舉 (用於 Contextual Bandit)
# ═══════════════════════════════════════════════════════════════════════════
class MarketContext:
    """
    市場狀態分類

    用於 Contextual Bandit，根據市場狀態選擇不同的參數策略:
    - RANGING: 震盪市 → 緊密間距，高頻交易
    - TRENDING_UP: 上漲趨勢 → 偏多策略
    - TRENDING_DOWN: 下跌趨勢 → 偏空策略
    - HIGH_VOLATILITY: 高波動 → 寬鬆間距，避免被掃
    """
    RANGING = "ranging"              # 震盪
    TRENDING_UP = "trending_up"      # 上漲趨勢
    TRENDING_DOWN = "trending_down"  # 下跌趨勢
    HIGH_VOLATILITY = "high_vol"     # 高波動

    # 每種市場狀態的推薦 arm 索引
    RECOMMENDED_ARMS = {
        RANGING: [0, 1, 2, 3],        # 緊密型，適合震盪
        TRENDING_UP: [4, 5],           # 平衡型
        TRENDING_DOWN: [4, 5],         # 平衡型
        HIGH_VOLATILITY: [6, 7, 8, 9]  # 寬鬆型，適合高波動
    }


@dataclass
class ParameterArm:
    """參數組合 (一個 Arm)"""
    gamma: float                # GLFT 風險係數
    grid_spacing: float         # 補倉間距
    take_profit_spacing: float  # 止盈間距

    def __hash__(self):
        return hash((self.gamma, self.grid_spacing, self.take_profit_spacing))

    def __str__(self):
        return f"γ={self.gamma:.2f}/GS={self.grid_spacing*100:.1f}%/TP={self.take_profit_spacing*100:.1f}%"


class UCBBanditOptimizer:
    """
    UCB Bandit 參數優化器 (增強版)

    基於論文: TradeBot: Bandit learning for hyper-parameters optimization

    增強功能:
    1. 冷啟動預載 - 首次運行使用歷史最佳參數，避免初期隨機探索的損失
    2. Contextual Bandit - 根據市場狀態 (趨勢/震盪/高波動) 選擇不同策略
    3. Thompson Sampling - 支持連續參數空間探索，通過參數擾動發現更優解
    4. 改進 Reward - 加入 Max Drawdown 懲罰和勝率獎勵

    AS 網格特點:
    - 每次只掛一買一賣
    - 成交後跟隨價格重新掛單
    - 對向倍投 (持倉多時反向加倍)
    """

    # AS 網格專用參數組合空間
    DEFAULT_ARMS = [
        # 緊密型 (高頻交易，手續費敏感) - 適合震盪市
        ParameterArm(gamma=0.05, grid_spacing=0.003, take_profit_spacing=0.003),
        ParameterArm(gamma=0.05, grid_spacing=0.004, take_profit_spacing=0.004),
        # 非對稱型 (止盈小於補倉，適合震盪)
        ParameterArm(gamma=0.08, grid_spacing=0.005, take_profit_spacing=0.003),
        ParameterArm(gamma=0.08, grid_spacing=0.006, take_profit_spacing=0.004),
        # 平衡型 - 適合趨勢市
        ParameterArm(gamma=0.10, grid_spacing=0.006, take_profit_spacing=0.004),
        ParameterArm(gamma=0.10, grid_spacing=0.008, take_profit_spacing=0.005),
        # 寬鬆型 (低頻交易) - 適合高波動
        ParameterArm(gamma=0.12, grid_spacing=0.008, take_profit_spacing=0.006),
        ParameterArm(gamma=0.12, grid_spacing=0.010, take_profit_spacing=0.006),
        # 高波動適應型
        ParameterArm(gamma=0.15, grid_spacing=0.010, take_profit_spacing=0.008),
        ParameterArm(gamma=0.15, grid_spacing=0.012, take_profit_spacing=0.008),
    ]

    def __init__(self, config: BanditConfig = None):
        self.config = config or BanditConfig()
        self.arms = self.DEFAULT_ARMS.copy()

        # 每個 arm 的獎勵歷史 (滑動窗口)
        self.rewards: Dict[int, deque] = {
            i: deque(maxlen=self.config.window_size)
            for i in range(len(self.arms))
        }

        # 追蹤狀態
        self.current_arm_idx: int = 0
        self.total_pulls: int = 0
        self.pull_counts: Dict[int, int] = {i: 0 for i in range(len(self.arms))}

        # 交易追蹤 (用於計算 reward)
        self.pending_trades: List[Dict] = []
        self.trade_count_since_update: int = 0

        # 統計
        self.best_arm_history: List[int] = []
        self.cumulative_reward: float = 0

        # === 新增: Contextual Bandit 狀態 ===
        self.current_context: str = MarketContext.RANGING
        self.price_history: deque = deque(maxlen=100)  # 價格歷史 (用於計算市場狀態)

        # 每個 context 的獨立統計
        self.context_rewards: Dict[str, Dict[int, deque]] = {
            ctx: {i: deque(maxlen=self.config.window_size) for i in range(len(self.arms))}
            for ctx in [MarketContext.RANGING, MarketContext.TRENDING_UP,
                       MarketContext.TRENDING_DOWN, MarketContext.HIGH_VOLATILITY]
        }
        self.context_pulls: Dict[str, Dict[int, int]] = {
            ctx: {i: 0 for i in range(len(self.arms))}
            for ctx in [MarketContext.RANGING, MarketContext.TRENDING_UP,
                       MarketContext.TRENDING_DOWN, MarketContext.HIGH_VOLATILITY]
        }

        # === 新增: Thompson Sampling 狀態 ===
        # Beta 分布參數 (alpha, beta) 用於每個 arm
        self.thompson_alpha: Dict[int, float] = {
            i: self.config.thompson_prior_alpha for i in range(len(self.arms))
        }
        self.thompson_beta: Dict[int, float] = {
            i: self.config.thompson_prior_beta for i in range(len(self.arms))
        }

        # === 新增: 動態生成的 arm (Thompson Sampling 探索) ===
        self.dynamic_arm: Optional[ParameterArm] = None
        self.dynamic_arm_reward: float = 0

        # === 冷啟動初始化 ===
        if self.config.cold_start_enabled:
            self._cold_start_init()

        logger.info(f"[Bandit] 增強版初始化完成，共 {len(self.arms)} 個參數組合")
        logger.info(f"[Bandit] 功能: 冷啟動={self.config.cold_start_enabled}, "
                   f"Contextual={self.config.contextual_enabled}, "
                   f"Thompson={self.config.thompson_enabled}")

    def _cold_start_init(self):
        """
        冷啟動初始化

        設計理念:
        - 首次運行時，避免從頭探索造成的損失
        - 使用歷史回測得出的最佳參數作為起點
        - 為推薦的 arm 預設一些正向獎勵
        """
        # 設定初始 arm 為歷史最佳
        self.current_arm_idx = self.config.cold_start_arm_idx

        # 為推薦 arm 預載一些虛擬正向獎勵 (給予信任)
        recommended_arms = [4, 5]  # 平衡型
        for arm_idx in recommended_arms:
            self.rewards[arm_idx].append(0.5)  # 預設正向獎勵
            self.pull_counts[arm_idx] = 1
            self.total_pulls += 1

        logger.info(f"[Bandit] 冷啟動: 初始 arm={self.current_arm_idx}, "
                   f"預載 arms={recommended_arms}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Contextual Bandit: 市場狀態檢測
    # ═══════════════════════════════════════════════════════════════════════════

    def update_price(self, price: float):
        """更新價格歷史 (用於計算市場狀態)"""
        self.price_history.append(price)

    def detect_market_context(self) -> str:
        """
        檢測當前市場狀態

        基於:
        1. 波動率 (標準差 / 均值)
        2. 趨勢 (線性回歸斜率)

        Returns:
            MarketContext 枚舉值
        """
        if not self.config.contextual_enabled:
            return MarketContext.RANGING

        if len(self.price_history) < self.config.volatility_lookback:
            return self.current_context  # 數據不足，保持當前狀態

        prices = list(self.price_history)

        # 計算波動率
        recent_prices = prices[-self.config.volatility_lookback:]
        volatility = np.std(recent_prices) / np.mean(recent_prices)

        # 檢查高波動
        if volatility > self.config.high_volatility_threshold:
            self.current_context = MarketContext.HIGH_VOLATILITY
            return self.current_context

        # 計算趨勢 (簡單線性回歸)
        if len(prices) >= self.config.trend_lookback:
            trend_prices = prices[-self.config.trend_lookback:]
            x = np.arange(len(trend_prices))
            slope = np.polyfit(x, trend_prices, 1)[0]
            trend_pct = slope / np.mean(trend_prices)

            if trend_pct > self.config.trend_threshold:
                self.current_context = MarketContext.TRENDING_UP
            elif trend_pct < -self.config.trend_threshold:
                self.current_context = MarketContext.TRENDING_DOWN
            else:
                self.current_context = MarketContext.RANGING
        else:
            self.current_context = MarketContext.RANGING

        return self.current_context

    # ═══════════════════════════════════════════════════════════════════════════
    # Thompson Sampling: 連續空間探索
    # ═══════════════════════════════════════════════════════════════════════════

    def _thompson_sample(self) -> int:
        """
        Thompson Sampling 選擇 arm

        從每個 arm 的 Beta 分布中採樣，選擇採樣值最高的 arm
        """
        samples = []
        for i in range(len(self.arms)):
            # 從 Beta 分布採樣
            sample = np.random.beta(self.thompson_alpha[i], self.thompson_beta[i])
            samples.append(sample)

        return int(np.argmax(samples))

    def _generate_dynamic_arm(self) -> Optional[ParameterArm]:
        """
        基於最佳 arm 生成動態參數組合 (參數擾動)

        設計理念:
        - 找到目前最佳的 arm
        - 在其參數附近進行小幅擾動
        - 探索可能更優的連續參數空間
        """
        if not self.config.thompson_enabled:
            return None

        # 找到最佳 arm
        best_idx = self._get_best_arm()
        best_arm = self.arms[best_idx]

        # 參數擾動
        perturbation = self.config.param_perturbation

        # 隨機擾動方向
        gamma_delta = np.random.uniform(-perturbation, perturbation) * best_arm.gamma
        gs_delta = np.random.uniform(-perturbation, perturbation) * best_arm.grid_spacing
        tp_delta = np.random.uniform(-perturbation, perturbation) * best_arm.take_profit_spacing

        # 生成新參數 (確保在合理範圍內)
        new_gamma = max(0.01, min(0.3, best_arm.gamma + gamma_delta))
        new_gs = max(0.002, min(0.02, best_arm.grid_spacing + gs_delta))
        new_tp = max(0.002, min(0.015, best_arm.take_profit_spacing + tp_delta))

        # 確保 tp < gs
        if new_tp >= new_gs:
            new_tp = new_gs * 0.7

        return ParameterArm(gamma=new_gamma, grid_spacing=new_gs, take_profit_spacing=new_tp)

    # ═══════════════════════════════════════════════════════════════════════════
    # 改進的 Reward 計算
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_reward(self, pnls: List[float]) -> float:
        """
        計算改進的 Reward

        公式:
        reward = sharpe_ratio - mdd_penalty * max_drawdown + win_rate_bonus * win_rate

        改進點:
        1. Sharpe-like 指標 (平均收益 / 波動)
        2. Max Drawdown 懲罰 (避免大回撤)
        3. 勝率獎勵 (鼓勵穩定獲利)
        """
        if not pnls:
            return 0

        # 1. Sharpe-like reward
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if np.std(pnls) > 0 else 0.001
        sharpe = mean_pnl / std_pnl

        # 2. Max Drawdown 計算
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # 正規化 MDD (相對於總收益)
        total_pnl = sum(pnls)
        mdd_ratio = max_drawdown / abs(total_pnl) if total_pnl != 0 else 0
        mdd_penalty = self.config.mdd_penalty_weight * mdd_ratio

        # 3. 勝率獎勵
        win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
        win_bonus = self.config.win_rate_bonus * (win_rate - 0.5)  # 50% 為基準

        # 綜合 reward
        reward = sharpe - mdd_penalty + win_bonus

        return reward

    def _update_thompson(self, reward: float):
        """
        更新 Thompson Sampling 的 Beta 分布參數

        將 reward 轉換為 0-1 之間的成功/失敗概率
        """
        arm_idx = self.current_arm_idx

        # 將 reward 正規化到 [0, 1]
        # 使用 sigmoid 轉換
        prob_success = 1 / (1 + np.exp(-reward))

        # 更新 Beta 分布參數
        self.thompson_alpha[arm_idx] += prob_success
        self.thompson_beta[arm_idx] += (1 - prob_success)

    # ═══════════════════════════════════════════════════════════════════════════
    # 核心方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_current_params(self) -> ParameterArm:
        """獲取當前選擇的參數"""
        # 如果有動態 arm 且表現良好，使用它
        if self.dynamic_arm and self.dynamic_arm_reward > 0:
            return self.dynamic_arm
        return self.arms[self.current_arm_idx]

    def select_arm(self) -> int:
        """
        選擇 arm (融合 UCB + Contextual + Thompson)

        優先順序:
        1. Contextual: 根據市場狀態限制候選 arms
        2. Thompson: 有一定概率使用 Thompson Sampling
        3. UCB: 默認使用 UCB 算法
        """
        # 確保每個 arm 都至少被試過 min_pulls_per_arm 次
        for i in range(len(self.arms)):
            if self.pull_counts[i] < self.config.min_pulls_per_arm:
                return i

        # Contextual: 獲取當前市場狀態推薦的 arms
        if self.config.contextual_enabled:
            context = self.detect_market_context()
            recommended = MarketContext.RECOMMENDED_ARMS.get(context, list(range(len(self.arms))))
        else:
            recommended = list(range(len(self.arms)))

        # Thompson Sampling: 有 30% 概率使用
        if self.config.thompson_enabled and np.random.random() < 0.3:
            thompson_choice = self._thompson_sample()
            # 如果 Thompson 選擇在推薦列表中，使用它
            if thompson_choice in recommended:
                return thompson_choice

        # UCB 算法 (只在推薦的 arms 中選擇)
        ucb_values = []
        for i in range(len(self.arms)):
            if i not in recommended:
                ucb_values.append(float('-inf'))
                continue

            rewards = list(self.rewards[i])
            if not rewards:
                ucb_values.append(float('inf'))
                continue

            mean_reward = np.mean(rewards)
            confidence = self.config.exploration_factor * np.sqrt(
                2 * np.log(self.total_pulls + 1) / len(rewards)
            )
            ucb_values.append(mean_reward + confidence)

        return int(np.argmax(ucb_values))

    def record_trade(self, pnl: float, side: str):
        """
        記錄交易結果

        Args:
            pnl: 交易盈虧
            side: 'long' 或 'short'
        """
        if not self.config.enabled:
            return

        self.pending_trades.append({
            'pnl': pnl,
            'side': side,
            'arm_idx': self.current_arm_idx,
            'context': self.current_context,
            'timestamp': time.time()
        })
        self.trade_count_since_update += 1

        # 每 N 筆交易更新一次
        if self.trade_count_since_update >= self.config.update_interval:
            self._update_and_select()

    def _update_and_select(self):
        """更新獎勵並選擇新的 arm"""
        if not self.pending_trades:
            return

        pnls = [t['pnl'] for t in self.pending_trades]

        # 使用改進的 reward 計算
        reward = self._calculate_reward(pnls)

        # 更新當前 arm 的獎勵
        arm_idx = self.current_arm_idx
        self.rewards[arm_idx].append(reward)
        self.pull_counts[arm_idx] += 1
        self.total_pulls += 1
        self.cumulative_reward += sum(pnls)

        # 更新 Contextual 統計
        if self.config.contextual_enabled:
            context = self.pending_trades[0].get('context', MarketContext.RANGING)
            self.context_rewards[context][arm_idx].append(reward)
            self.context_pulls[context][arm_idx] += 1

        # 更新 Thompson Sampling 參數
        if self.config.thompson_enabled:
            self._update_thompson(reward)

        # 選擇下一個 arm
        new_arm_idx = self.select_arm()
        if new_arm_idx != self.current_arm_idx:
            old_params = self.arms[self.current_arm_idx]
            new_params = self.arms[new_arm_idx]
            logger.info(f"[Bandit] 切換參數: {old_params} → {new_params} "
                       f"(context={self.current_context})")
            self.current_arm_idx = new_arm_idx

        # 偶爾嘗試動態生成的 arm (10% 概率)
        if self.config.thompson_enabled and np.random.random() < 0.1:
            self.dynamic_arm = self._generate_dynamic_arm()
            if self.dynamic_arm:
                logger.info(f"[Bandit] 動態探索: {self.dynamic_arm}")

        # 記錄最佳 arm
        self.best_arm_history.append(self._get_best_arm())

        # 清空待處理交易
        self.pending_trades = []
        self.trade_count_since_update = 0

    def _get_best_arm(self) -> int:
        """獲取目前表現最好的 arm"""
        best_idx = 0
        best_mean = float('-inf')

        for i in range(len(self.arms)):
            rewards = list(self.rewards[i])
            if rewards:
                mean = np.mean(rewards)
                if mean > best_mean:
                    best_mean = mean
                    best_idx = i

        return best_idx

    def get_stats(self) -> Dict:
        """獲取優化器統計 (增強版)"""
        best_idx = self._get_best_arm()
        arm_stats = []

        for i in range(len(self.arms)):
            rewards = list(self.rewards[i])
            arm_stats.append({
                'arm': str(self.arms[i]),
                'pulls': self.pull_counts[i],
                'mean_reward': np.mean(rewards) if rewards else 0,
                'is_current': i == self.current_arm_idx,
                'is_best': i == best_idx,
                'thompson_alpha': self.thompson_alpha[i],
                'thompson_beta': self.thompson_beta[i]
            })

        return {
            'enabled': self.config.enabled,
            'total_pulls': self.total_pulls,
            'current_arm': str(self.arms[self.current_arm_idx]),
            'best_arm': str(self.arms[best_idx]),
            'cumulative_reward': self.cumulative_reward,
            'current_context': self.current_context,
            'dynamic_arm': str(self.dynamic_arm) if self.dynamic_arm else None,
            'arm_stats': arm_stats
        }

    def to_dict(self) -> dict:
        """序列化狀態 (增強版)"""
        return {
            'current_arm_idx': self.current_arm_idx,
            'total_pulls': self.total_pulls,
            'pull_counts': dict(self.pull_counts),
            'rewards': {k: list(v) for k, v in self.rewards.items()},
            'cumulative_reward': self.cumulative_reward,
            'current_context': self.current_context,
            'thompson_alpha': dict(self.thompson_alpha),
            'thompson_beta': dict(self.thompson_beta),
            'context_pulls': {ctx: dict(pulls) for ctx, pulls in self.context_pulls.items()}
        }

    def load_state(self, state: dict):
        """載入狀態 (增強版)"""
        if not state:
            return
        self.current_arm_idx = state.get('current_arm_idx', 0)
        self.total_pulls = state.get('total_pulls', 0)
        self.pull_counts = {int(k): v for k, v in state.get('pull_counts', {}).items()}
        self.cumulative_reward = state.get('cumulative_reward', 0)
        self.current_context = state.get('current_context', MarketContext.RANGING)

        # 載入 rewards
        saved_rewards = state.get('rewards', {})
        for k, v in saved_rewards.items():
            idx = int(k)
            if idx in self.rewards:
                self.rewards[idx] = deque(v, maxlen=self.config.window_size)

        # 載入 Thompson 參數
        saved_alpha = state.get('thompson_alpha', {})
        for k, v in saved_alpha.items():
            self.thompson_alpha[int(k)] = v

        saved_beta = state.get('thompson_beta', {})
        for k, v in saved_beta.items():
            self.thompson_beta[int(k)] = v

        # 載入 context 統計
        saved_context_pulls = state.get('context_pulls', {})
        for ctx, pulls in saved_context_pulls.items():
            if ctx in self.context_pulls:
                self.context_pulls[ctx] = {int(k): v for k, v in pulls.items()}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         DGT 動態網格邊界管理                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class DGTConfig:
    """
    DGT (Dynamic Grid Trading) 配置

    注意: 此功能對 AS 高頻網格 (買一賣一) 效果有限
    AS 網格是跟隨價格的，沒有固定邊界概念
    保留此配置是為了未來可能的多層網格支援
    """
    enabled: bool = False              # 預設關閉 (AS 網格不需要)
    reset_threshold: float = 0.05      # 價格偏離多少觸發重置 (5%)
    profit_reinvest_ratio: float = 0.5 # 利潤再投資比例
    boundary_buffer: float = 0.02      # 邊界緩衝 (2%)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "reset_threshold": self.reset_threshold,
            "profit_reinvest_ratio": self.profit_reinvest_ratio,
            "boundary_buffer": self.boundary_buffer
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DGTConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DGTBoundaryManager:
    """
    DGT 動態邊界管理器

    基於論文: Dynamic Grid Trading Strategy: From Zero Expectation to Market Outperformance

    功能:
    1. 追蹤網格的上下邊界
    2. 當價格突破邊界時自動重置
    3. 將套利利潤再投資
    """

    def __init__(self, config: DGTConfig = None):
        self.config = config or DGTConfig()

        # 每個交易對的邊界狀態
        self.boundaries: Dict[str, Dict] = {}

        # 累積利潤 (用於再投資)
        self.accumulated_profits: Dict[str, float] = {}

        # 重置計數
        self.reset_counts: Dict[str, int] = {}

    def initialize_boundary(self, symbol: str, center_price: float, grid_spacing: float, num_grids: int = 10):
        """
        初始化網格邊界

        Args:
            symbol: 交易對
            center_price: 中心價格
            grid_spacing: 網格間距
            num_grids: 網格數量 (上下各半)
        """
        half_grids = num_grids // 2

        # 幾何網格計算
        upper = center_price * ((1 + grid_spacing) ** half_grids)
        lower = center_price * ((1 - grid_spacing) ** half_grids)

        self.boundaries[symbol] = {
            'center': center_price,
            'upper': upper,
            'lower': lower,
            'grid_spacing': grid_spacing,
            'num_grids': num_grids,
            'initialized_at': time.time(),
            'last_reset': time.time()
        }

        self.accumulated_profits[symbol] = 0
        self.reset_counts[symbol] = 0

        logger.info(f"[DGT] {symbol} 邊界初始化: {lower:.4f} ~ {upper:.4f} (中心: {center_price:.4f})")

    def check_and_reset(self, symbol: str, current_price: float, realized_pnl: float = 0) -> Tuple[bool, Optional[Dict]]:
        """
        檢查是否需要重置邊界

        Args:
            symbol: 交易對
            current_price: 當前價格
            realized_pnl: 自上次重置以來的已實現盈虧

        Returns:
            (是否重置, 新邊界資訊)
        """
        if not self.config.enabled:
            return False, None

        if symbol not in self.boundaries:
            return False, None

        boundary = self.boundaries[symbol]
        upper = boundary['upper']
        lower = boundary['lower']

        # 檢查是否突破邊界
        breach_upper = current_price >= upper * (1 - self.config.boundary_buffer)
        breach_lower = current_price <= lower * (1 + self.config.boundary_buffer)

        if not (breach_upper or breach_lower):
            return False, None

        # 累積利潤
        self.accumulated_profits[symbol] += realized_pnl

        # 重置邊界
        old_center = boundary['center']
        new_center = current_price

        # 根據突破方向決定策略
        if breach_upper:
            # 上破: 回收本金 + 部分利潤再投資
            reinvest = self.accumulated_profits[symbol] * self.config.profit_reinvest_ratio
            logger.info(f"[DGT] {symbol} 上破重置: {old_center:.4f} → {new_center:.4f}, 再投資: {reinvest:.2f}")
        else:
            # 下破: 持有幣，用套利利潤當新本金
            reinvest = self.accumulated_profits[symbol]
            logger.info(f"[DGT] {symbol} 下破重置: {old_center:.4f} → {new_center:.4f}, 累積利潤: {reinvest:.2f}")

        # 更新邊界
        self.initialize_boundary(
            symbol,
            new_center,
            boundary['grid_spacing'],
            boundary['num_grids']
        )

        self.reset_counts[symbol] += 1
        self.accumulated_profits[symbol] = 0  # 重置累積利潤

        return True, {
            'old_center': old_center,
            'new_center': new_center,
            'direction': 'upper' if breach_upper else 'lower',
            'reinvest_amount': reinvest,
            'reset_count': self.reset_counts[symbol]
        }

    def get_adjusted_spacing(self, symbol: str, base_spacing: float) -> float:
        """
        根據距離邊界的位置調整間距

        靠近邊界時縮小間距，增加交易機會
        """
        if symbol not in self.boundaries:
            return base_spacing

        boundary = self.boundaries[symbol]
        center = boundary['center']
        upper = boundary['upper']
        lower = boundary['lower']

        # 這個功能暫時不啟用，保持原始間距
        return base_spacing

    def get_boundary_info(self, symbol: str) -> Optional[Dict]:
        """獲取邊界資訊"""
        if symbol not in self.boundaries:
            return None

        boundary = self.boundaries[symbol]
        return {
            'center': boundary['center'],
            'upper': boundary['upper'],
            'lower': boundary['lower'],
            'reset_count': self.reset_counts.get(symbol, 0),
            'accumulated_profit': self.accumulated_profits.get(symbol, 0)
        }

    def get_stats(self) -> Dict:
        """獲取統計"""
        return {
            'enabled': self.config.enabled,
            'symbols': list(self.boundaries.keys()),
            'total_resets': sum(self.reset_counts.values()),
            'total_accumulated_profit': sum(self.accumulated_profits.values()),
            'boundaries': {
                symbol: self.get_boundary_info(symbol)
                for symbol in self.boundaries
            }
        }


class FundingRateManager:
    """
    Funding Rate 管理器

    功能:
    - 獲取當前 funding rate
    - 計算持倉偏向調整
    - funding rate > 0: 多付空收 → 偏向持有空頭
    - funding rate < 0: 空付多收 → 偏向持有多頭
    """

    def __init__(self, exchange):
        self.exchange = exchange
        self.funding_rates: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self.update_interval = 60  # 每 60 秒更新一次

    def update_funding_rate(self, symbol: str) -> float:
        """更新並返回 funding rate"""
        now = time.time()

        # 檢查是否需要更新
        if symbol in self.last_update:
            if now - self.last_update[symbol] < self.update_interval:
                return self.funding_rates.get(symbol, 0)

        try:
            # 獲取 funding rate
            funding_info = self.exchange.fetch_funding_rate(symbol)
            rate = float(funding_info.get('fundingRate', 0) or 0)

            self.funding_rates[symbol] = rate
            self.last_update[symbol] = now

            logger.info(f"[Funding] {symbol} funding rate: {rate*100:.4f}%")
            return rate

        except Exception as e:
            logger.error(f"[Funding] 獲取 {symbol} funding rate 失敗: {e}")
            return self.funding_rates.get(symbol, 0)

    def get_position_bias(self, symbol: str, config: 'MaxEnhancement') -> Tuple[float, float]:
        """
        根據 funding rate 計算持倉偏向

        Returns:
            (long_bias, short_bias): 多空偏向係數
            - 1.0 表示正常
            - > 1.0 表示增加該方向
            - < 1.0 表示減少該方向
        """
        if not config.is_feature_enabled('funding_rate'):
            return 1.0, 1.0

        rate = self.funding_rates.get(symbol, 0)

        # funding rate 不顯著時不調整
        if abs(rate) < config.funding_rate_threshold:
            return 1.0, 1.0

        bias = config.funding_rate_position_bias

        if rate > 0:
            # 多付空收 → 減少多頭，增加空頭
            long_bias = 1.0 - bias
            short_bias = 1.0 + bias
        else:
            # 空付多收 → 增加多頭，減少空頭
            long_bias = 1.0 + bias
            short_bias = 1.0 - bias

        return long_bias, short_bias


class GLFTController:
    """
    GLFT (Guéant-Lehalle-Fernandez-Tapia) 庫存控制器

    核心概念:
    - 根據庫存偏離調整報價
    - γ (gamma): 風險厭惡係數，越大越厭惡庫存偏離
    - 庫存偏離 = (多頭持倉 - 空頭持倉) / (多頭持倉 + 空頭持倉)

    報價偏移公式 (簡化版):
    - bid_skew = -inventory_ratio × base_spread × γ
    - ask_skew = +inventory_ratio × base_spread × γ
    """

    def calculate_inventory_ratio(self, long_pos: float, short_pos: float) -> float:
        """
        計算庫存比例

        Returns:
            -1.0 到 1.0 之間的值
            - 1.0: 全部多頭
            - -1.0: 全部空頭
            - 0.0: 多空平衡
        """
        total = long_pos + short_pos
        if total <= 0:
            return 0.0

        return (long_pos - short_pos) / total

    def calculate_spread_skew(
        self,
        long_pos: float,
        short_pos: float,
        base_spread: float,
        config: 'MaxEnhancement'
    ) -> Tuple[float, float]:
        """
        計算報價偏移

        Args:
            long_pos: 多頭持倉
            short_pos: 空頭持倉
            base_spread: 基礎間距
            config: MAX 配置

        Returns:
            (bid_skew, ask_skew): 買賣價偏移比例
        """
        if not config.is_feature_enabled('glft'):
            return 0.0, 0.0

        inventory_ratio = self.calculate_inventory_ratio(long_pos, short_pos)

        # 偏移量 = 庫存比例 × 基礎間距 × γ
        skew = inventory_ratio * base_spread * config.gamma

        # bid_skew: 庫存多時壓低買價 (減少買入)
        # ask_skew: 庫存多時提高賣價 (增加賣出)
        bid_skew = -skew
        ask_skew = skew

        return bid_skew, ask_skew

    def adjust_order_quantity(
        self,
        base_qty: float,
        side: str,
        long_pos: float,
        short_pos: float,
        config: 'MaxEnhancement'
    ) -> float:
        """
        根據庫存調整訂單數量

        當庫存偏離時:
        - 減少偏離方向的補倉數量
        - 增加反方向的補倉數量
        """
        if not config.is_feature_enabled('glft'):
            return base_qty

        inventory_ratio = self.calculate_inventory_ratio(long_pos, short_pos)

        # 調整係數
        if side == 'long':
            # 庫存偏多時減少多頭補倉
            adjust = 1.0 - inventory_ratio * config.gamma
        else:
            # 庫存偏空時減少空頭補倉
            adjust = 1.0 + inventory_ratio * config.gamma

        # 限制調整範圍 [0.5, 1.5]
        adjust = max(0.5, min(1.5, adjust))

        return base_qty * adjust


class DynamicGridManager:
    """
    動態網格管理器

    功能:
    - 根據 ATR/波動率調整網格間距
    - 高波動 → 大間距 (避免頻繁觸發)
    - 低波動 → 小間距 (捕捉小波動)
    """

    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.atr_cache: Dict[str, float] = {}
        self.last_calc_time: Dict[str, float] = {}
        self.calc_interval = 60  # 每 60 秒重算一次

    def update_price(self, symbol: str, price: float):
        """更新價格歷史"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)

        self.price_history[symbol].append({
            'price': price,
            'time': time.time()
        })

    def calculate_atr(self, symbol: str, config: 'MaxEnhancement') -> float:
        """
        計算 ATR (Average True Range) 百分比

        簡化版: 使用價格變動的標準差作為波動率估計
        """
        now = time.time()

        # 檢查是否需要重算
        if symbol in self.last_calc_time:
            if now - self.last_calc_time[symbol] < self.calc_interval:
                return self.atr_cache.get(symbol, 0.005)

        history = self.price_history.get(symbol, deque())
        if len(history) < config.volatility_lookback:
            return 0.005  # 預設 0.5%

        # 取最近 N 個價格
        recent_prices = [h['price'] for h in list(history)[-config.volatility_lookback:]]

        # 計算收益率
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)

        if not returns:
            return 0.005

        # 波動率 = 收益率標準差 × ATR 乘數
        volatility = np.std(returns) * config.atr_multiplier

        # 限制範圍
        volatility = max(config.min_spacing, min(config.max_spacing, volatility))

        self.atr_cache[symbol] = volatility
        self.last_calc_time[symbol] = now

        return volatility

    def get_dynamic_spacing(
        self,
        symbol: str,
        base_take_profit: float,
        base_grid_spacing: float,
        config: 'MaxEnhancement'
    ) -> Tuple[float, float]:
        """
        獲取動態調整後的間距

        Returns:
            (take_profit_spacing, grid_spacing)
        """
        if not config.is_feature_enabled('dynamic_grid'):
            return base_take_profit, base_grid_spacing

        atr = self.calculate_atr(symbol, config)

        # 動態止盈間距 = ATR × 0.5 (比補倉小)
        dynamic_tp = atr * 0.5
        dynamic_tp = max(config.min_spacing, min(config.max_spacing * 0.6, dynamic_tp))

        # 動態補倉間距 = ATR
        dynamic_gs = atr
        dynamic_gs = max(config.min_spacing * 1.5, min(config.max_spacing, dynamic_gs))

        # 確保止盈 < 補倉
        if dynamic_tp >= dynamic_gs:
            dynamic_tp = dynamic_gs * 0.6

        return dynamic_tp, dynamic_gs


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         領先指標系統 (取代滯後 ATR)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class LeadingIndicatorConfig:
    """
    領先指標配置

    核心理念:
    - ATR/波動率是「滯後指標」: 價格已經動了才知道
    - 領先指標: 在價格大幅波動「之前」就能察覺

    使用的領先因子:
    1. Order Flow Imbalance (OFI) - 訂單流失衡，反映買賣壓力
    2. Volume Surge - 成交量突增，預示即將突破
    3. Spread Expansion - 買賣價差擴大，預示流動性變差/波動即將放大
    """
    enabled: bool = True

    # === OFI (Order Flow Imbalance) ===
    ofi_enabled: bool = True
    ofi_lookback: int = 20                  # OFI 計算回看期
    ofi_threshold: float = 0.6              # OFI > 此值 = 強烈買壓 or 賣壓

    # === Volume Surge ===
    volume_enabled: bool = True
    volume_lookback: int = 50               # 成交量回看期
    volume_surge_threshold: float = 2.0     # 成交量 > 平均 × 此值 = 異常放量

    # === Spread Analysis ===
    spread_enabled: bool = True
    spread_lookback: int = 30               # 價差回看期
    spread_surge_threshold: float = 1.5     # 價差 > 平均 × 此值 = 流動性下降

    # === 綜合信號 ===
    min_signals_for_action: int = 2         # 至少 N 個信號同時觸發才調整

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "ofi_enabled": self.ofi_enabled,
            "ofi_lookback": self.ofi_lookback,
            "ofi_threshold": self.ofi_threshold,
            "volume_enabled": self.volume_enabled,
            "volume_lookback": self.volume_lookback,
            "volume_surge_threshold": self.volume_surge_threshold,
            "spread_enabled": self.spread_enabled,
            "spread_lookback": self.spread_lookback,
            "spread_surge_threshold": self.spread_surge_threshold,
            "min_signals_for_action": self.min_signals_for_action
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LeadingIndicatorConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LeadingIndicatorManager:
    """
    領先指標管理器

    取代滯後的 ATR 波動率指標，使用三種領先因子：

    1. Order Flow Imbalance (OFI)
       公式: OFI = (買量 - 賣量) / (買量 + 賣量)
       範圍: -1 到 1
       - OFI > 0.6: 強烈買壓，價格可能上漲
       - OFI < -0.6: 強烈賣壓，價格可能下跌
       應用: 偏向調整方向

    2. Volume Surge (成交量突增)
       公式: Volume Ratio = 當前成交量 / 平均成交量
       信號: Ratio > 2.0 表示異常放量
       應用: 放大間距，避免被突破掃掉

    3. Spread Expansion (價差擴大)
       公式: Spread Ratio = 當前價差 / 平均價差
       信號: Ratio > 1.5 表示流動性下降
       應用: 預期波動放大，調整間距
    """

    def __init__(self, config: LeadingIndicatorConfig = None):
        self.config = config or LeadingIndicatorConfig()

        # 數據存儲 (每個交易對)
        self.trade_history: Dict[str, deque] = {}      # 成交記錄
        self.spread_history: Dict[str, deque] = {}     # 價差記錄
        self.ofi_history: Dict[str, deque] = {}        # OFI 歷史

        # 當前狀態
        self.current_ofi: Dict[str, float] = {}
        self.current_volume_ratio: Dict[str, float] = {}
        self.current_spread_ratio: Dict[str, float] = {}

        # 信號狀態
        self.active_signals: Dict[str, List[str]] = {}

        logger.info("[LeadingIndicator] 領先指標管理器初始化完成")

    def _ensure_symbol_data(self, symbol: str):
        """確保交易對數據結構存在"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=500)
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=200)
        if symbol not in self.ofi_history:
            self.ofi_history[symbol] = deque(maxlen=100)

    def record_trade(self, symbol: str, price: float, quantity: float, side: str):
        """
        記錄成交 (從 WebSocket 接收)

        Args:
            symbol: 交易對
            price: 成交價
            quantity: 成交量
            side: 'buy' 或 'sell'
        """
        if not self.config.enabled:
            return

        self._ensure_symbol_data(symbol)

        self.trade_history[symbol].append({
            'time': time.time(),
            'price': price,
            'quantity': quantity,
            'side': side,
            'value': price * quantity
        })

    def update_spread(self, symbol: str, bid: float, ask: float):
        """
        更新買賣價差

        Args:
            symbol: 交易對
            bid: 最佳買價
            ask: 最佳賣價
        """
        if not self.config.enabled or bid <= 0 or ask <= 0:
            return

        self._ensure_symbol_data(symbol)

        mid_price = (bid + ask) / 2
        spread_bps = (ask - bid) / mid_price * 10000  # 基點

        self.spread_history[symbol].append({
            'time': time.time(),
            'bid': bid,
            'ask': ask,
            'spread_bps': spread_bps
        })

    def calculate_ofi(self, symbol: str) -> float:
        """
        計算 Order Flow Imbalance

        Returns:
            OFI 值 (-1 到 1)
            - 正值: 買壓強
            - 負值: 賣壓強
            - 接近 0: 平衡
        """
        if symbol not in self.trade_history:
            return 0.0

        trades = list(self.trade_history[symbol])
        if len(trades) < self.config.ofi_lookback:
            return 0.0

        # 取最近 N 筆
        recent = trades[-self.config.ofi_lookback:]

        buy_volume = sum(t['value'] for t in recent if t['side'] == 'buy')
        sell_volume = sum(t['value'] for t in recent if t['side'] == 'sell')

        total = buy_volume + sell_volume
        if total <= 0:
            return 0.0

        ofi = (buy_volume - sell_volume) / total

        # 更新狀態
        self.current_ofi[symbol] = ofi
        self.ofi_history[symbol].append({
            'time': time.time(),
            'ofi': ofi,
            'buy_vol': buy_volume,
            'sell_vol': sell_volume
        })

        return ofi

    def calculate_volume_ratio(self, symbol: str) -> float:
        """
        計算成交量比率

        Returns:
            當前成交量 / 平均成交量
            - > 2.0: 異常放量
            - < 0.5: 異常縮量
        """
        if symbol not in self.trade_history:
            return 1.0

        trades = list(self.trade_history[symbol])
        if len(trades) < self.config.volume_lookback:
            return 1.0

        # 按分鐘聚合
        now = time.time()
        recent_minute = [t['value'] for t in trades if now - t['time'] < 60]
        historical = trades[-self.config.volume_lookback:]

        current_volume = sum(recent_minute)
        avg_volume_per_trade = np.mean([t['value'] for t in historical])
        expected_volume = avg_volume_per_trade * max(1, len(recent_minute))

        if expected_volume <= 0:
            return 1.0

        ratio = current_volume / expected_volume

        self.current_volume_ratio[symbol] = ratio
        return ratio

    def calculate_spread_ratio(self, symbol: str) -> float:
        """
        計算價差比率

        Returns:
            當前價差 / 平均價差
            - > 1.5: 價差擴大 (流動性下降)
            - < 0.8: 價差收窄 (流動性良好)
        """
        if symbol not in self.spread_history:
            return 1.0

        spreads = list(self.spread_history[symbol])
        if len(spreads) < self.config.spread_lookback:
            return 1.0

        current_spread = spreads[-1]['spread_bps']
        avg_spread = np.mean([s['spread_bps'] for s in spreads[-self.config.spread_lookback:]])

        if avg_spread <= 0:
            return 1.0

        ratio = current_spread / avg_spread

        self.current_spread_ratio[symbol] = ratio
        return ratio

    def get_signals(self, symbol: str) -> Tuple[List[str], Dict[str, float]]:
        """
        獲取當前活躍信號

        Returns:
            (信號列表, 指標值字典)
        """
        if not self.config.enabled:
            return [], {}

        signals = []
        values = {}

        # 計算各指標
        ofi = self.calculate_ofi(symbol)
        volume_ratio = self.calculate_volume_ratio(symbol)
        spread_ratio = self.calculate_spread_ratio(symbol)

        values = {
            'ofi': ofi,
            'volume_ratio': volume_ratio,
            'spread_ratio': spread_ratio
        }

        # OFI 信號
        if self.config.ofi_enabled:
            if ofi > self.config.ofi_threshold:
                signals.append('OFI_BUY_PRESSURE')
            elif ofi < -self.config.ofi_threshold:
                signals.append('OFI_SELL_PRESSURE')

        # Volume 信號
        if self.config.volume_enabled:
            if volume_ratio > self.config.volume_surge_threshold:
                signals.append('VOLUME_SURGE')

        # Spread 信號
        if self.config.spread_enabled:
            if spread_ratio > self.config.spread_surge_threshold:
                signals.append('SPREAD_EXPANSION')

        self.active_signals[symbol] = signals
        return signals, values

    def get_spacing_adjustment(self, symbol: str, base_spacing: float) -> Tuple[float, str]:
        """
        根據領先指標計算間距調整

        核心邏輯:
        - 正常情況: 保持基礎間距
        - Volume Surge 或 Spread Expansion: 放大間距 (預期波動)
        - OFI 極端值: 可能方向性移動，謹慎調整

        Args:
            symbol: 交易對
            base_spacing: 基礎間距

        Returns:
            (調整後間距, 原因說明)
        """
        if not self.config.enabled:
            return base_spacing, "領先指標關閉"

        signals, values = self.get_signals(symbol)

        # 沒有信號，保持原樣
        if not signals:
            return base_spacing, "正常"

        # 計算調整係數
        adjustment = 1.0
        reasons = []

        # Volume Surge: 放大間距 20-50%
        if 'VOLUME_SURGE' in signals:
            vol_ratio = values.get('volume_ratio', 1.0)
            # 成交量越大，間距放大越多
            vol_adj = min(1.5, 1.0 + (vol_ratio - 2.0) * 0.1)
            adjustment = max(adjustment, vol_adj)
            reasons.append(f"放量×{vol_ratio:.1f}")

        # Spread Expansion: 放大間距 20-40%
        if 'SPREAD_EXPANSION' in signals:
            spread_ratio = values.get('spread_ratio', 1.0)
            spread_adj = min(1.4, 1.0 + (spread_ratio - 1.5) * 0.2)
            adjustment = max(adjustment, spread_adj)
            reasons.append(f"價差擴{spread_ratio:.1f}x")

        # OFI 極端: 小幅放大間距 (防止被單邊掃)
        if 'OFI_BUY_PRESSURE' in signals or 'OFI_SELL_PRESSURE' in signals:
            ofi = abs(values.get('ofi', 0))
            ofi_adj = 1.0 + ofi * 0.2  # 最多放大 20%
            adjustment = max(adjustment, ofi_adj)
            direction = "買" if values.get('ofi', 0) > 0 else "賣"
            reasons.append(f"{direction}壓OFI={ofi:.2f}")

        # 限制最大調整
        adjustment = min(adjustment, 1.8)  # 最多放大 80%

        adjusted_spacing = base_spacing * adjustment
        reason = " + ".join(reasons) if reasons else "正常"

        return adjusted_spacing, reason

    def get_direction_bias(self, symbol: str) -> Tuple[float, float, str]:
        """
        根據 OFI 計算方向偏向

        當 OFI 顯示強烈方向時，調整多空偏好

        Returns:
            (long_bias, short_bias, reason)
            - bias > 1.0: 增加該方向
            - bias < 1.0: 減少該方向
        """
        if not self.config.enabled or not self.config.ofi_enabled:
            return 1.0, 1.0, ""

        ofi = self.current_ofi.get(symbol, 0)

        if abs(ofi) < self.config.ofi_threshold * 0.5:
            return 1.0, 1.0, "OFI平衡"

        # OFI > 0: 買壓強，偏向做多
        # OFI < 0: 賣壓強，偏向做空
        bias_strength = abs(ofi) * 0.3  # 最多 30% 偏向

        if ofi > 0:
            long_bias = 1.0 + bias_strength
            short_bias = 1.0 - bias_strength * 0.5
            reason = f"買壓+{ofi:.2f}"
        else:
            long_bias = 1.0 - bias_strength * 0.5
            short_bias = 1.0 + bias_strength
            reason = f"賣壓{ofi:.2f}"

        return long_bias, short_bias, reason

    def should_pause_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        判斷是否應該暫停交易

        極端情況:
        - Volume 異常 (> 4x) + Spread 異常 (> 2x) = 暫停
        - 可能是大消息或閃崩

        Returns:
            (是否暫停, 原因)
        """
        if not self.config.enabled:
            return False, ""

        signals, values = self.get_signals(symbol)

        volume_ratio = values.get('volume_ratio', 1.0)
        spread_ratio = values.get('spread_ratio', 1.0)

        # 極端條件
        if volume_ratio > 4.0 and spread_ratio > 2.0:
            return True, f"極端波動 (Vol={volume_ratio:.1f}x, Spread={spread_ratio:.1f}x)"

        # 單一極端
        if volume_ratio > 6.0:
            return True, f"異常放量 (Vol={volume_ratio:.1f}x)"

        if spread_ratio > 3.0:
            return True, f"流動性枯竭 (Spread={spread_ratio:.1f}x)"

        return False, ""

    def get_stats(self, symbol: str = None) -> Dict:
        """獲取統計資訊"""
        if symbol:
            signals, values = self.get_signals(symbol)
            return {
                'symbol': symbol,
                'enabled': self.config.enabled,
                'ofi': values.get('ofi', 0),
                'volume_ratio': values.get('volume_ratio', 1.0),
                'spread_ratio': values.get('spread_ratio', 1.0),
                'active_signals': signals,
                'trade_count': len(self.trade_history.get(symbol, [])),
                'spread_count': len(self.spread_history.get(symbol, []))
            }

        return {
            'enabled': self.config.enabled,
            'symbols': list(self.trade_history.keys()),
            'config': self.config.to_dict()
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              配置類                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class SymbolConfig:
    """單一交易對配置"""
    symbol: str = "XRPUSDC"
    ccxt_symbol: str = "XRP/USDC:USDC"
    enabled: bool = True

    # 基礎策略參數 (會被動態調整)
    take_profit_spacing: float = 0.004
    grid_spacing: float = 0.006
    initial_quantity: float = 3
    leverage: int = 20

    # 持倉控制 - 動態倍數 (基於 initial_quantity 自動計算)
    # position_limit = initial_quantity × limit_multiplier (觸發止盈加倍)
    # position_threshold = initial_quantity × threshold_multiplier (觸發裝死模式)
    limit_multiplier: float = 5.0       # 5單後止盈加倍
    threshold_multiplier: float = 20.0  # 20單後裝死

    @property
    def coin_name(self) -> str:
        return self.ccxt_symbol.split('/')[0]

    @property
    def contract_type(self) -> str:
        return self.ccxt_symbol.split('/')[1].split(':')[0]

    @property
    def ws_symbol(self) -> str:
        return f"{self.coin_name.lower()}{self.contract_type.lower()}"

    @property
    def position_limit(self) -> float:
        """動態計算持倉限制 (止盈加倍閾值)"""
        return self.initial_quantity * self.limit_multiplier

    @property
    def position_threshold(self) -> float:
        """動態計算持倉閾值 (裝死模式閾值)"""
        return self.initial_quantity * self.threshold_multiplier

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ccxt_symbol": self.ccxt_symbol,
            "enabled": self.enabled,
            "take_profit_spacing": self.take_profit_spacing,
            "grid_spacing": self.grid_spacing,
            "initial_quantity": self.initial_quantity,
            "leverage": self.leverage,
            "limit_multiplier": self.limit_multiplier,
            "threshold_multiplier": self.threshold_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SymbolConfig':
        # 兼容舊配置：如果有舊的 position_threshold/position_limit，轉換為倍數
        if "position_threshold" in data and "threshold_multiplier" not in data:
            qty = data.get("initial_quantity", 3)
            if qty > 0:
                data["threshold_multiplier"] = data["position_threshold"] / qty
            del data["position_threshold"]
        if "position_limit" in data and "limit_multiplier" not in data:
            qty = data.get("initial_quantity", 3)
            if qty > 0:
                data["limit_multiplier"] = data["position_limit"] / qty
            del data["position_limit"]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskConfig:
    """風控配置"""
    enabled: bool = True
    margin_threshold: float = 0.5
    trailing_start_profit: float = 5.0
    trailing_drawdown_pct: float = 0.10
    trailing_min_drawdown: float = 2.0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "margin_threshold": self.margin_threshold,
            "trailing_start_profit": self.trailing_start_profit,
            "trailing_drawdown_pct": self.trailing_drawdown_pct,
            "trailing_min_drawdown": self.trailing_min_drawdown
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RiskConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GlobalConfig:
    """全局配置"""
    api_key: str = ""
    api_secret: str = ""
    websocket_url: str = "wss://fstream.binance.com/ws"
    sync_interval: float = 30.0
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    max_enhancement: MaxEnhancement = field(default_factory=MaxEnhancement)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    dgt: DGTConfig = field(default_factory=DGTConfig)
    leading_indicator: LeadingIndicatorConfig = field(default_factory=LeadingIndicatorConfig)
    # Story 1.4: 偵測舊版配置是否包含明文 API
    legacy_api_detected: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        # 保存 API 凭证到配置文件
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "websocket_url": self.websocket_url,
            "sync_interval": self.sync_interval,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "risk": self.risk.to_dict(),
            "max_enhancement": self.max_enhancement.to_dict(),
            "bandit": self.bandit.to_dict(),
            "dgt": self.dgt.to_dict(),
            "leading_indicator": self.leading_indicator.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GlobalConfig':
        # 从配置文件加载 API 凭证
        config = cls(
            api_key=data.get("api_key", ""),
            api_secret=data.get("api_secret", ""),
            websocket_url=data.get("websocket_url", "wss://fstream.binance.com/ws"),
            sync_interval=data.get("sync_interval", 30.0),
            legacy_api_detected=False
        )
        for k, v in data.get("symbols", {}).items():
            config.symbols[k] = SymbolConfig.from_dict(v)
        if "risk" in data:
            config.risk = RiskConfig.from_dict(data["risk"])
        if "max_enhancement" in data:
            config.max_enhancement = MaxEnhancement.from_dict(data["max_enhancement"])
        if "bandit" in data:
            config.bandit = BanditConfig.from_dict(data["bandit"])
        if "dgt" in data:
            config.dgt = DGTConfig.from_dict(data["dgt"])
        if "leading_indicator" in data:
            config.leading_indicator = LeadingIndicatorConfig.from_dict(data["leading_indicator"])
        return config

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print("[green]配置已保存[/]")

    @classmethod
    def load(cls) -> 'GlobalConfig':
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return cls.from_dict(json.load(f))
        return cls()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              交易狀態                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class SymbolState:
    """單一交易對狀態"""
    symbol: str
    latest_price: float = 0
    best_bid: float = 0
    best_ask: float = 0
    long_position: float = 0
    short_position: float = 0
    unrealized_pnl: float = 0
    buy_long_orders: float = 0
    sell_long_orders: float = 0
    buy_short_orders: float = 0
    sell_short_orders: float = 0
    tracking_active: bool = False
    peak_pnl: float = 0
    current_pnl: float = 0
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=5))
    total_trades: int = 0
    total_profit: float = 0

    # 裝死模式狀態
    long_dead_mode: bool = False
    short_dead_mode: bool = False

    # 網格價格追蹤
    last_grid_price_long: float = 0
    last_grid_price_short: float = 0

    # MAX 增強狀態
    current_funding_rate: float = 0
    dynamic_take_profit: float = 0
    dynamic_grid_spacing: float = 0
    inventory_ratio: float = 0

    # 領先指標狀態
    leading_ofi: float = 0               # Order Flow Imbalance
    leading_volume_ratio: float = 1.0    # 成交量比率
    leading_spread_ratio: float = 1.0    # 價差比率
    leading_signals: List[str] = field(default_factory=list)  # 活躍信號


@dataclass
class AccountBalance:
    """單一帳戶餘額"""
    currency: str = "USDC"
    wallet_balance: float = 0      # 錢包餘額
    available_balance: float = 0   # 可用餘額
    unrealized_pnl: float = 0      # 未實現盈虧
    margin_used: float = 0         # 已用保證金

    @property
    def equity(self) -> float:
        """權益 = 錢包餘額 + 未實現盈虧"""
        return self.wallet_balance + self.unrealized_pnl

    @property
    def margin_ratio(self) -> float:
        """保證金使用率"""
        if self.equity <= 0:
            return 0
        return self.margin_used / self.equity


@dataclass
class GlobalState:
    """全局狀態"""
    running: bool = False
    connected: bool = False
    start_time: Optional[datetime] = None

    # 分帳戶餘額 (USDC / USDT)
    accounts: Dict[str, AccountBalance] = field(default_factory=lambda: {
        "USDC": AccountBalance(currency="USDC"),
        "USDT": AccountBalance(currency="USDT")
    })

    # 舊的全局字段 (保持向後兼容)
    total_equity: float = 0
    free_balance: float = 0
    margin_usage: float = 0
    total_unrealized_pnl: float = 0

    symbols: Dict[str, SymbolState] = field(default_factory=dict)
    total_trades: int = 0
    total_profit: float = 0

    # 追蹤止盈狀態
    trailing_active: Dict[str, bool] = field(default_factory=dict)
    peak_pnl: Dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0

    # 雙向減倉冷卻
    last_reduce_time: Dict[str, float] = field(default_factory=dict)

    def get_account(self, currency: str) -> AccountBalance:
        """獲取指定幣種帳戶"""
        if currency not in self.accounts:
            self.accounts[currency] = AccountBalance(currency=currency)
        return self.accounts[currency]

    def update_totals(self):
        """更新總計數據"""
        self.total_equity = sum(acc.equity for acc in self.accounts.values())
        self.free_balance = sum(acc.available_balance for acc in self.accounts.values())
        self.total_unrealized_pnl = sum(acc.unrealized_pnl for acc in self.accounts.values())
        if self.total_equity > 0:
            total_margin = sum(acc.margin_used for acc in self.accounts.values())
            self.margin_usage = total_margin / self.total_equity


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              回測管理器                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BacktestManager:
    """回測管理器 - 簡化版，直接輸入交易對符號"""

    def __init__(self):
        self.data_dir = DATA_DIR

    def get_data_path(self, symbol_raw: str) -> Path:
        """獲取數據路徑"""
        return self.data_dir / f"futures/um/daily/klines/{symbol_raw}/1m"

    def get_available_dates(self, symbol_raw: str) -> List[str]:
        """獲取可用日期"""
        path = self.get_data_path(symbol_raw)
        if not path.exists():
            return []

        dates = []
        for f in path.glob(f"{symbol_raw}-1m-*.csv"):
            try:
                parts = f.stem.split('-')
                if len(parts) >= 5:
                    date_str = f"{parts[2]}-{parts[3]}-{parts[4]}"
                    dates.append(date_str)
            except Exception:
                pass

        return sorted(dates)

    def load_data(self, symbol_raw: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """載入歷史數據"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_data = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            path = self.get_data_path(symbol_raw) / f"{symbol_raw}-1m-{date_str}.csv"

            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if 'open_time' in df.columns:
                        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    all_data.append(df)
                except Exception as e:
                    console.print(f"[yellow]載入 {date_str} 失敗: {e}[/]")

            current += timedelta(days=1)

        if not all_data:
            return None

        full_df = pd.concat(all_data, ignore_index=True)
        return full_df.sort_values('open_time').reset_index(drop=True)

    def download_data(self, symbol_raw: str, ccxt_symbol: str, start_date: str, end_date: str) -> bool:
        """下載歷史數據"""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })

            # 轉換為 ccxt 格式 (不帶 :USDC)
            fetch_symbol = ccxt_symbol.split(":")[0]  # "XRP/USDC"

            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            total_bars = 0
            days = (end - start).days + 1

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"下載 {symbol_raw}...", total=days)
                current = start

                while current <= end:
                    date_str = current.strftime("%Y-%m-%d")
                    output_path = self.get_data_path(symbol_raw) / f"{symbol_raw}-1m-{date_str}.csv"

                    if not output_path.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        since = int(datetime(current.year, current.month, current.day).timestamp() * 1000)
                        until = since + 24 * 60 * 60 * 1000

                        try:
                            ohlcv = exchange.fetch_ohlcv(fetch_symbol, "1m", since=since, limit=1500)
                            if ohlcv:
                                ohlcv = [bar for bar in ohlcv if bar[0] < until]
                                df = pd.DataFrame(ohlcv, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
                                df.to_csv(output_path, index=False)
                                total_bars += len(df)
                        except Exception as e:
                            console.print(f"[red]{date_str}: {e}[/]")

                    current += timedelta(days=1)
                    progress.update(task, advance=1)

            console.print(f"[green]下載完成: {total_bars:,} 條數據[/]")
            return True

        except Exception as e:
            console.print(f"[red]下載失敗: {e}[/]")
            return False

    def run_backtest(self, config: SymbolConfig, df: pd.DataFrame) -> dict:
        """
        執行回測

        同步實盤邏輯:
        1. position_threshold (裝死模式): 持倉超過此值不補倉，只掛特殊止盈
        2. position_limit (止盈加倍): 持倉超過此值或對側超過 threshold，止盈數量加倍
        """
        balance = 1000.0
        max_equity = balance

        long_positions = []
        short_positions = []
        trades = []
        equity_curve = []

        order_value = config.initial_quantity * df['close'].iloc[0]
        leverage = config.leverage
        fee_pct = 0.0004

        last_long_price = df['close'].iloc[0]
        last_short_price = df['close'].iloc[0]

        # 持倉控制參數
        position_threshold = config.position_threshold
        position_limit = config.position_limit

        for _, row in df.iterrows():
            price = row['close']

            # 計算當前持倉量
            long_position = sum(p["qty"] for p in long_positions)
            short_position = sum(p["qty"] for p in short_positions)

            # === 多頭網格 (使用 GridStrategy 統一邏輯) ===
            long_decision = GridStrategy.get_grid_decision(
                price=last_long_price,
                my_position=long_position,
                opposite_position=short_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='long'
            )

            sell_price = long_decision['tp_price']
            long_tp_qty = long_decision['tp_qty']
            long_dead_mode = long_decision['dead_mode']
            buy_price = long_decision['entry_price'] if long_decision['entry_price'] else last_long_price * (1 - config.grid_spacing)

            if not long_dead_mode:
                # 【正常模式】補倉邏輯
                if price <= buy_price:
                    qty = order_value / price
                    margin = (qty * price) / leverage
                    fee = qty * price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        long_positions.append({"price": price, "qty": qty, "margin": margin})
                        last_long_price = price

            # 止盈邏輯 (兩種模式都執行)
            if price >= sell_price and long_positions:
                # 根據止盈數量決定平倉多少
                remaining_tp = long_tp_qty
                while long_positions and remaining_tp > 0:
                    pos = long_positions[0]
                    if pos["qty"] <= remaining_tp:
                        # 全部平倉
                        long_positions.pop(0)
                        gross_pnl = (price - pos["price"]) * pos["qty"]
                        fee = pos["qty"] * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += pos["margin"] + net_pnl
                        trades.append({"pnl": net_pnl, "type": "long"})
                        remaining_tp -= pos["qty"]
                    else:
                        # 部分平倉
                        close_ratio = remaining_tp / pos["qty"]
                        close_qty = remaining_tp
                        close_margin = pos["margin"] * close_ratio
                        gross_pnl = (price - pos["price"]) * close_qty
                        fee = close_qty * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += close_margin + net_pnl
                        trades.append({"pnl": net_pnl, "type": "long"})
                        pos["qty"] -= close_qty
                        pos["margin"] -= close_margin
                        remaining_tp = 0
                last_long_price = price

            # === 空頭網格 (使用 GridStrategy 統一邏輯) ===
            short_decision = GridStrategy.get_grid_decision(
                price=last_short_price,
                my_position=short_position,
                opposite_position=long_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='short'
            )

            cover_price = short_decision['tp_price']
            short_tp_qty = short_decision['tp_qty']
            short_dead_mode = short_decision['dead_mode']
            sell_short_price = short_decision['entry_price'] if short_decision['entry_price'] else last_short_price * (1 + config.grid_spacing)

            if not short_dead_mode:
                # 【正常模式】補倉邏輯
                if price >= sell_short_price:
                    qty = order_value / price
                    margin = (qty * price) / leverage
                    fee = qty * price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        short_positions.append({"price": price, "qty": qty, "margin": margin})
                        last_short_price = price

            # 止盈邏輯 (兩種模式都執行)
            if price <= cover_price and short_positions:
                # 根據止盈數量決定平倉多少
                remaining_tp = short_tp_qty
                while short_positions and remaining_tp > 0:
                    pos = short_positions[0]
                    if pos["qty"] <= remaining_tp:
                        # 全部平倉
                        short_positions.pop(0)
                        gross_pnl = (pos["price"] - price) * pos["qty"]
                        fee = pos["qty"] * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += pos["margin"] + net_pnl
                        trades.append({"pnl": net_pnl, "type": "short"})
                        remaining_tp -= pos["qty"]
                    else:
                        # 部分平倉
                        close_ratio = remaining_tp / pos["qty"]
                        close_qty = remaining_tp
                        close_margin = pos["margin"] * close_ratio
                        gross_pnl = (pos["price"] - price) * close_qty
                        fee = close_qty * price * fee_pct
                        net_pnl = gross_pnl - fee
                        balance += close_margin + net_pnl
                        trades.append({"pnl": net_pnl, "type": "short"})
                        pos["qty"] -= close_qty
                        pos["margin"] -= close_margin
                        remaining_tp = 0
                last_short_price = price

            # 計算淨值
            unrealized = sum((price - p["price"]) * p["qty"] for p in long_positions)
            unrealized += sum((p["price"] - price) * p["qty"] for p in short_positions)
            equity = balance + unrealized
            max_equity = max(max_equity, equity)
            equity_curve.append(equity)

        # 計算結果
        final_price = df['close'].iloc[-1]
        unrealized_pnl = sum((final_price - p["price"]) * p["qty"] for p in long_positions)
        unrealized_pnl += sum((p["price"] - final_price) * p["qty"] for p in short_positions)

        realized_pnl = sum(t["pnl"] for t in trades)
        final_equity = balance + unrealized_pnl

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] < 0]

        return {
            "final_equity": final_equity,
            "return_pct": (final_equity - 1000) / 1000,
            "max_drawdown": 1 - (min(equity_curve) / max_equity) if equity_curve else 0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "trades_count": len(trades),
            "win_rate": len(winning) / len(trades) if trades else 0,
            "profit_factor": sum(t["pnl"] for t in winning) / abs(sum(t["pnl"] for t in losing)) if losing else float('inf')
        }

    def optimize_params(self, config: SymbolConfig, df: pd.DataFrame, progress_callback=None) -> List[dict]:
        """優化參數"""
        results = []

        take_profits = [0.002, 0.003, 0.004, 0.005, 0.006]
        grid_spacings = [0.004, 0.006, 0.008, 0.01, 0.012]

        valid_combos = [(tp, gs) for tp in take_profits for gs in grid_spacings if tp < gs]
        total = len(valid_combos)

        for i, (tp, gs) in enumerate(valid_combos):
            test_config = SymbolConfig(
                symbol=config.symbol,
                ccxt_symbol=config.ccxt_symbol,
                take_profit_spacing=tp,
                grid_spacing=gs,
                initial_quantity=config.initial_quantity,
                leverage=config.leverage
            )

            result = self.run_backtest(test_config, df)
            result["take_profit_spacing"] = tp
            result["grid_spacing"] = gs
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        results.sort(key=lambda x: x["return_pct"], reverse=True)
        return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              交易所擴展                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CustomExchange(ccxt.binance):
    def fetch(self, url, method='GET', headers=None, body=None):
        if headers is None:
            headers = {}
        return super().fetch(url, method, headers, body)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAX 網格交易機器人                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MaxGridBot:
    """MAX 版本網格機器人 - 整合學術模型增強功能"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.state = GlobalState()

        for symbol, sym_cfg in config.symbols.items():
            if sym_cfg.enabled:
                self.state.symbols[sym_cfg.ccxt_symbol] = SymbolState(symbol=sym_cfg.ccxt_symbol)

        self.exchange: Optional[CustomExchange] = None
        self.listen_key: Optional[str] = None
        self.tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self.precisions: Dict[str, dict] = {}
        self.last_sync_time = 0
        self.last_order_times: Dict[str, float] = {}

        # MAX 增強模組
        self.funding_manager: Optional[FundingRateManager] = None
        self.glft_controller = GLFTController()
        self.dynamic_grid_manager = DynamicGridManager()

        # 學習模組 (Bandit + DGT)
        self.bandit_optimizer = UCBBanditOptimizer(config.bandit)
        self.dgt_manager = DGTBoundaryManager(config.dgt)

        # 領先指標系統 (取代滯後 ATR)
        self.leading_indicator = LeadingIndicatorManager(config.leading_indicator)

        logger.info(f"[MAX] 初始化完成 - Bandit: {config.bandit.enabled}, Leading: {config.leading_indicator.enabled}")

    def _init_exchange(self):
        self.exchange = CustomExchange({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "options": {"defaultType": "future"}
        })
        self.exchange.load_markets(reload=False)

        # 初始化 funding manager
        self.funding_manager = FundingRateManager(self.exchange)

        markets = self.exchange.fetch_markets()
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue

            try:
                symbol_info = next(m for m in markets if m["symbol"] == sym_config.ccxt_symbol)
                price_prec = symbol_info["precision"]["price"]
                self.precisions[sym_config.ccxt_symbol] = {
                    "price": int(abs(math.log10(price_prec))) if isinstance(price_prec, float) else price_prec,
                    "amount": int(abs(math.log10(symbol_info["precision"]["amount"]))) if isinstance(symbol_info["precision"]["amount"], float) else symbol_info["precision"]["amount"],
                    "min_amount": symbol_info["limits"]["amount"]["min"]
                }
            except Exception as e:
                logger.error(f"獲取 {sym_config.ccxt_symbol} 精度失敗: {e}")

    def _check_hedge_mode(self):
        for sym_config in self.config.symbols.values():
            if sym_config.enabled:
                try:
                    mode = self.exchange.fetch_position_mode(symbol=sym_config.ccxt_symbol)
                    if not mode['hedged']:
                        self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                        break
                except Exception:
                    pass

    def _get_listen_key(self) -> str:
        response = self.exchange.fapiPrivatePostListenKey()
        return response.get("listenKey")

    def sync_all(self):
        self._sync_positions()
        self._sync_orders()
        self._sync_account()
        self._sync_funding_rates()

    def _sync_funding_rates(self):
        """同步所有交易對的 funding rate"""
        if not self.funding_manager:
            return

        for sym_config in self.config.symbols.values():
            if sym_config.enabled:
                rate = self.funding_manager.update_funding_rate(sym_config.ccxt_symbol)
                sym_state = self.state.symbols.get(sym_config.ccxt_symbol)
                if sym_state:
                    sym_state.current_funding_rate = rate

    def _sync_positions(self):
        try:
            positions = self.exchange.fetch_positions(params={'type': 'future'})

            for sym_state in self.state.symbols.values():
                sym_state.long_position = 0
                sym_state.short_position = 0
                sym_state.unrealized_pnl = 0

            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.state.symbols:
                    contracts = pos.get('contracts', 0)
                    side = pos.get('side')
                    pnl = float(pos.get('unrealizedPnl', 0) or 0)

                    if side == 'long':
                        self.state.symbols[symbol].long_position = contracts
                    elif side == 'short':
                        self.state.symbols[symbol].short_position = abs(contracts)

                    self.state.symbols[symbol].unrealized_pnl += pnl

        except Exception as e:
            logger.error(f"同步持倉失敗: {e}")

    def _sync_orders(self):
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            symbol = sym_config.ccxt_symbol

            try:
                orders = self.exchange.fetch_open_orders(symbol=symbol)
                state = self.state.symbols.get(symbol)
                if not state:
                    continue

                state.buy_long_orders = 0
                state.sell_long_orders = 0
                state.buy_short_orders = 0
                state.sell_short_orders = 0

                for order in orders:
                    qty = abs(float(order.get('info', {}).get('origQty', 0)))
                    side = order.get('side')
                    pos_side = order.get('info', {}).get('positionSide')

                    if side == 'buy' and pos_side == 'LONG':
                        state.buy_long_orders += qty
                    elif side == 'sell' and pos_side == 'LONG':
                        state.sell_long_orders += qty
                    elif side == 'buy' and pos_side == 'SHORT':
                        state.buy_short_orders += qty
                    elif side == 'sell' and pos_side == 'SHORT':
                        state.sell_short_orders += qty
            except Exception as e:
                logger.error(f"同步 {symbol} 掛單失敗: {e}")

    def _sync_account(self):
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})

            # 分別更新 USDC 和 USDT 帳戶
            for currency in ['USDC', 'USDT']:
                total = float(balance.get('total', {}).get(currency, 0) or 0)
                free = float(balance.get('free', {}).get(currency, 0) or 0)

                acc = self.state.get_account(currency)
                acc.wallet_balance = total
                acc.available_balance = free
                acc.margin_used = total - free if total > free else 0

                # 計算該幣種的未實現盈虧
                unrealized = 0
                for sym_state in self.state.symbols.values():
                    if currency in sym_state.symbol:
                        unrealized += sym_state.unrealized_pnl
                acc.unrealized_pnl = unrealized

            # 更新總計
            self.state.update_totals()

            self._check_trailing_stop()
        except Exception as e:
            logger.error(f"同步帳戶失敗: {e}")

    def _check_trailing_stop(self):
        """保證金追蹤止盈邏輯"""
        risk = self.config.risk

        if not risk.enabled:
            return

        if self.state.margin_usage < risk.margin_threshold:
            self.state.trailing_active.clear()
            self.state.peak_pnl.clear()
            return

        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue

            ccxt_symbol = sym_config.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_symbol)
            if not sym_state:
                continue

            current_pnl = sym_state.unrealized_pnl

            if self.state.trailing_active.get(ccxt_symbol, False):
                peak = self.state.peak_pnl.get(ccxt_symbol, 0)
                if current_pnl > peak:
                    self.state.peak_pnl[ccxt_symbol] = current_pnl
                    logger.info(f"[追蹤止盈] {sym_config.symbol} 新高: {current_pnl:.2f}U")

                peak = self.state.peak_pnl.get(ccxt_symbol, 0)
                drawdown = peak - current_pnl

                trigger = max(risk.trailing_min_drawdown, peak * risk.trailing_drawdown_pct)

                if drawdown >= trigger and peak > 0:
                    logger.info(f"[追蹤止盈] {sym_config.symbol} 觸發! 最高:{peak:.2f}, 當前:{current_pnl:.2f}, 回撤:{drawdown:.2f}")
                    self._close_symbol_positions(ccxt_symbol, sym_config)
                    self.state.trailing_active[ccxt_symbol] = False
                    self.state.peak_pnl[ccxt_symbol] = 0

            else:
                if current_pnl >= risk.trailing_start_profit:
                    self.state.trailing_active[ccxt_symbol] = True
                    self.state.peak_pnl[ccxt_symbol] = current_pnl
                    logger.info(f"[追蹤止盈] {sym_config.symbol} 開始追蹤! 浮盈: {current_pnl:.2f}U")

    def _close_symbol_positions(self, ccxt_symbol: str, sym_config: SymbolConfig):
        """平倉指定交易對"""
        try:
            sym_state = self.state.symbols.get(ccxt_symbol)
            if not sym_state:
                return

            self.cancel_orders_for_side(ccxt_symbol, 'long')
            self.cancel_orders_for_side(ccxt_symbol, 'short')

            if sym_state.long_position > 0:
                self.place_order(
                    ccxt_symbol, 'sell', 0, sym_state.long_position,
                    reduce_only=True, position_side='long', order_type='market'
                )
                logger.info(f"[追蹤止盈] {sym_config.symbol} 市價平多 {sym_state.long_position}")

            if sym_state.short_position > 0:
                self.place_order(
                    ccxt_symbol, 'buy', 0, sym_state.short_position,
                    reduce_only=True, position_side='short', order_type='market'
                )
                logger.info(f"[追蹤止盈] {sym_config.symbol} 市價平空 {sym_state.short_position}")

        except Exception as e:
            logger.error(f"[追蹤止盈] {sym_config.symbol} 平倉失敗: {e}")

    def place_order(self, symbol: str, side: str, price: float, quantity: float,
                    reduce_only: bool = False, position_side: str = None,
                    order_type: str = 'limit'):
        try:
            prec = self.precisions.get(symbol, {"price": 4, "amount": 0, "min_amount": 1})
            price = round(price, prec["price"])
            quantity = round(quantity, prec["amount"])
            quantity = max(quantity, prec["min_amount"])

            params = {'reduce_only': reduce_only}
            if position_side:
                params['positionSide'] = position_side.upper()

            if order_type == 'market':
                return self.exchange.create_order(symbol, 'market', side, quantity, params=params)
            else:
                return self.exchange.create_order(symbol, 'limit', side, quantity, price, params)
        except Exception as e:
            logger.error(f"下單失敗 {symbol}: {e}")
            return None

    def cancel_orders_for_side(self, symbol: str, position_side: str):
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            for order in orders:
                order_side = order.get('side')
                order_pos_side = order.get('info', {}).get('positionSide', 'BOTH')
                reduce_only = order.get('reduceOnly', False)

                should_cancel = False
                if position_side == 'long':
                    if (not reduce_only and order_side == 'buy' and order_pos_side == 'LONG') or \
                       (reduce_only and order_side == 'sell' and order_pos_side == 'LONG'):
                        should_cancel = True
                elif position_side == 'short':
                    if (not reduce_only and order_side == 'sell' and order_pos_side == 'SHORT') or \
                       (reduce_only and order_side == 'buy' and order_pos_side == 'SHORT'):
                        should_cancel = True

                if should_cancel:
                    self.exchange.cancel_order(order['id'], symbol)
        except Exception as e:
            logger.error(f"撤單失敗 {symbol}: {e}")

    def _get_dynamic_spacing(self, sym_config: SymbolConfig, sym_state: SymbolState) -> Tuple[float, float]:
        """
        獲取動態調整後的間距

        優先順序:
        1. 領先指標 (OFI, Volume, Spread) - 預測未來波動
        2. 動態網格 (ATR) - 滯後指標作為備用
        3. GLFT 偏移 - 庫存控制
        """
        max_cfg = self.config.max_enhancement
        ccxt_symbol = sym_config.ccxt_symbol

        # 基礎間距
        base_take_profit = sym_config.take_profit_spacing
        base_grid_spacing = sym_config.grid_spacing

        # === 1. 領先指標調整 (優先於 ATR) ===
        leading_reason = ""
        leading_signals = []
        leading_values = {}

        if self.config.leading_indicator.enabled:
            # 先獲取信號 (避免重複計算)
            leading_signals, leading_values = self.leading_indicator.get_signals(ccxt_symbol)

            # 更新狀態 (用於 UI 顯示)
            sym_state.leading_ofi = leading_values.get('ofi', 0)
            sym_state.leading_volume_ratio = leading_values.get('volume_ratio', 1.0)
            sym_state.leading_spread_ratio = leading_values.get('spread_ratio', 1.0)
            sym_state.leading_signals = leading_signals

            # 檢查是否應該暫停交易 (極端情況)
            should_pause, pause_reason = self.leading_indicator.should_pause_trading(ccxt_symbol)
            if should_pause:
                logger.warning(f"[LeadingIndicator] {sym_config.symbol} 暫停交易: {pause_reason}")
                # 極端情況下放大間距 100%，減少交易
                base_take_profit *= 2.0
                base_grid_spacing *= 2.0
                leading_reason = f"暫停:{pause_reason}"
            elif leading_signals:
                # 正常領先指標調整
                adjusted_spacing, leading_reason = self.leading_indicator.get_spacing_adjustment(
                    ccxt_symbol, base_grid_spacing
                )
                if adjusted_spacing != base_grid_spacing:
                    ratio = adjusted_spacing / base_grid_spacing
                    base_grid_spacing = adjusted_spacing
                    base_take_profit *= ratio  # 等比例調整止盈

        # === 2. 動態網格範圍 (ATR - 滯後指標) ===
        # 如果領先指標已經調整，則跳過 ATR
        if not leading_reason or leading_reason == "正常":
            take_profit, grid_spacing = self.dynamic_grid_manager.get_dynamic_spacing(
                ccxt_symbol,
                base_take_profit,
                base_grid_spacing,
                max_cfg
            )
        else:
            take_profit = base_take_profit
            grid_spacing = base_grid_spacing

        # === 3. GLFT 偏移 (根據庫存調整) ===
        bid_skew, ask_skew = self.glft_controller.calculate_spread_skew(
            sym_state.long_position,
            sym_state.short_position,
            grid_spacing,
            max_cfg
        )

        # 記錄到狀態
        sym_state.dynamic_take_profit = take_profit
        sym_state.dynamic_grid_spacing = grid_spacing
        sym_state.inventory_ratio = self.glft_controller.calculate_inventory_ratio(
            sym_state.long_position, sym_state.short_position
        )

        # 記錄領先指標原因 (用於日誌)
        if leading_reason and leading_reason != "正常":
            logger.debug(f"[LeadingIndicator] {sym_config.symbol} 間距調整: {leading_reason}")

        return take_profit, grid_spacing

    def _get_adjusted_quantity(
        self,
        sym_config: SymbolConfig,
        sym_state: SymbolState,
        side: str,
        is_take_profit: bool
    ) -> float:
        """
        獲取調整後的數量
        整合: Funding Rate 偏向 + GLFT 數量調整 + 原有邏輯
        """
        max_cfg = self.config.max_enhancement
        base_qty = sym_config.initial_quantity

        # 1. 原有邏輯: position_limit / position_threshold
        if is_take_profit:
            if side == 'long':
                if sym_state.long_position > sym_config.position_limit:
                    base_qty *= 2
                elif sym_state.short_position >= sym_config.position_threshold:
                    base_qty *= 2
            else:
                if sym_state.short_position > sym_config.position_limit:
                    base_qty *= 2
                elif sym_state.long_position >= sym_config.position_threshold:
                    base_qty *= 2

        # 2. GLFT 數量調整 (補倉時)
        if not is_take_profit:
            base_qty = self.glft_controller.adjust_order_quantity(
                base_qty, side,
                sym_state.long_position, sym_state.short_position,
                max_cfg
            )

        # 3. Funding Rate 偏向
        if self.funding_manager:
            long_bias, short_bias = self.funding_manager.get_position_bias(
                sym_config.ccxt_symbol, max_cfg
            )

            if side == 'long':
                base_qty *= long_bias
            else:
                base_qty *= short_bias

        return max(sym_config.initial_quantity * 0.5, base_qty)

    def _check_and_reduce_positions(self, sym_config: SymbolConfig, sym_state: SymbolState):
        """檢查並減倉 (雙向持倉過大時)"""
        REDUCE_COOLDOWN = 60

        ccxt_symbol = sym_config.ccxt_symbol
        local_threshold = sym_config.position_threshold * 0.8
        reduce_qty = sym_config.position_threshold * 0.1

        last_reduce = self.state.last_reduce_time.get(ccxt_symbol, 0)
        if time.time() - last_reduce < REDUCE_COOLDOWN:
            return

        if sym_state.long_position >= local_threshold and sym_state.short_position >= local_threshold:
            logger.info(f"[風控] {sym_config.symbol} 多空持倉均超過 {local_threshold}，開始雙向減倉")

            if sym_state.long_position > 0:
                self.place_order(ccxt_symbol, 'sell', 0, reduce_qty, True, 'long', 'market')
                logger.info(f"[風控] {sym_config.symbol} 市價平多 {reduce_qty}")

            if sym_state.short_position > 0:
                self.place_order(ccxt_symbol, 'buy', 0, reduce_qty, True, 'short', 'market')
                logger.info(f"[風控] {sym_config.symbol} 市價平空 {reduce_qty}")

            self.state.last_reduce_time[ccxt_symbol] = time.time()

    def _should_adjust_grid(self, sym_config: SymbolConfig, sym_state: SymbolState, side: str) -> bool:
        """檢查是否需要調整網格"""
        price = sym_state.latest_price
        deviation_threshold = sym_config.grid_spacing * 0.5

        if side == 'long':
            if sym_state.buy_long_orders <= 0 or sym_state.sell_long_orders <= 0:
                return True
            if sym_state.last_grid_price_long > 0:
                deviation = abs(price - sym_state.last_grid_price_long) / sym_state.last_grid_price_long
                return deviation >= deviation_threshold
            return True
        else:
            if sym_state.buy_short_orders <= 0 or sym_state.sell_short_orders <= 0:
                return True
            if sym_state.last_grid_price_short > 0:
                deviation = abs(price - sym_state.last_grid_price_short) / sym_state.last_grid_price_short
                return deviation >= deviation_threshold
            return True

    async def adjust_grid(self, ccxt_symbol: str):
        sym_config = None
        for cfg in self.config.symbols.values():
            if cfg.ccxt_symbol == ccxt_symbol and cfg.enabled:
                sym_config = cfg
                break

        if not sym_config:
            return

        sym_state = self.state.symbols.get(ccxt_symbol)
        if not sym_state:
            return

        price = sym_state.latest_price
        if price <= 0:
            return

        # === DGT 動態邊界管理 ===
        if self.config.dgt.enabled:
            # 初始化邊界 (首次)
            if ccxt_symbol not in self.dgt_manager.boundaries:
                self.dgt_manager.initialize_boundary(
                    ccxt_symbol, price, sym_config.grid_spacing, num_grids=10
                )

            # 檢查是否需要重置邊界
            accumulated = self.dgt_manager.accumulated_profits.get(ccxt_symbol, 0)
            reset, reset_info = self.dgt_manager.check_and_reset(ccxt_symbol, price, accumulated)
            if reset and reset_info:
                logger.info(f"[DGT] {sym_config.symbol} 邊界重置 #{reset_info['reset_count']}: "
                           f"{reset_info['direction']}破, 中心價 {reset_info['old_center']:.4f} → {reset_info['new_center']:.4f}")

        # === Bandit 參數應用 (AS 網格核心學習) ===
        if self.config.bandit.enabled:
            bandit_params = self.bandit_optimizer.get_current_params()
            # 動態覆蓋參數
            sym_config.grid_spacing = bandit_params.grid_spacing
            sym_config.take_profit_spacing = bandit_params.take_profit_spacing
            # 如果增強模式開啟，也調整 gamma
            if self.config.max_enhancement.all_enhancements_enabled:
                self.config.max_enhancement.gamma = bandit_params.gamma

        # 更新價格歷史 (for 動態網格)
        self.dynamic_grid_manager.update_price(ccxt_symbol, price)

        # 檢查並減倉
        self._check_and_reduce_positions(sym_config, sym_state)

        # 多頭
        if sym_state.long_position == 0:
            if time.time() - self.last_order_times.get(f"{ccxt_symbol}_long", 0) > 10:
                self.cancel_orders_for_side(ccxt_symbol, 'long')
                qty = self._get_adjusted_quantity(sym_config, sym_state, 'long', False)
                self.place_order(ccxt_symbol, 'buy', sym_state.best_bid, qty, False, 'long')
                self.last_order_times[f"{ccxt_symbol}_long"] = time.time()
                sym_state.last_grid_price_long = price
        else:
            if self._should_adjust_grid(sym_config, sym_state, 'long'):
                await self._place_grid(ccxt_symbol, sym_config, 'long')
                sym_state.last_grid_price_long = price

        # 空頭
        if sym_state.short_position == 0:
            if time.time() - self.last_order_times.get(f"{ccxt_symbol}_short", 0) > 10:
                self.cancel_orders_for_side(ccxt_symbol, 'short')
                qty = self._get_adjusted_quantity(sym_config, sym_state, 'short', False)
                self.place_order(ccxt_symbol, 'sell', sym_state.best_ask, qty, False, 'short')
                self.last_order_times[f"{ccxt_symbol}_short"] = time.time()
                sym_state.last_grid_price_short = price
        else:
            if self._should_adjust_grid(sym_config, sym_state, 'short'):
                await self._place_grid(ccxt_symbol, sym_config, 'short')
                sym_state.last_grid_price_short = price

    async def _place_grid(self, ccxt_symbol: str, sym_config: SymbolConfig, side: str):
        """
        掛出網格訂單 (MAX 版本)

        整合:
        1. 動態間距 (ATR)
        2. GLFT 庫存偏移
        3. Funding Rate 偏向
        """
        sym_state = self.state.symbols[ccxt_symbol]
        price = sym_state.latest_price

        # === 關鍵防護: 參考 Web 版本，防止無倉位時執行 reduce_only ===
        if side == 'long' and sym_state.long_position <= 0:
            logger.debug(f"[Grid] {sym_config.symbol} 多頭無倉位，跳過 _place_grid")
            return
        if side == 'short' and sym_state.short_position <= 0:
            logger.debug(f"[Grid] {sym_config.symbol} 空頭無倉位，跳過 _place_grid")
            return

        # 獲取動態調整後的間距
        take_profit_spacing, grid_spacing = self._get_dynamic_spacing(sym_config, sym_state)

        # 獲取調整後的數量
        tp_qty = self._get_adjusted_quantity(sym_config, sym_state, side, True)
        base_qty = self._get_adjusted_quantity(sym_config, sym_state, side, False)

        # === 使用 GridStrategy 統一計算邏輯 ===
        if side == 'long':
            my_position = sym_state.long_position
            opposite_position = sym_state.short_position
            dead_mode_flag = sym_state.long_dead_mode
            pending_tp_orders = sym_state.sell_long_orders
        else:
            my_position = sym_state.short_position
            opposite_position = sym_state.long_position
            dead_mode_flag = sym_state.short_dead_mode
            pending_tp_orders = sym_state.buy_short_orders

        # 使用 GridStrategy 判斷模式並計算價格
        is_dead = GridStrategy.is_dead_mode(my_position, sym_config.position_threshold)

        if is_dead:
            # === 裝死模式 ===
            if not dead_mode_flag:
                # 狀態轉換: 進入裝死模式
                if side == 'long':
                    sym_state.long_dead_mode = True
                else:
                    sym_state.short_dead_mode = True
                logger.info(f"[MAX] {sym_config.symbol} {side}頭進入裝死模式 (持倉:{my_position})")

            if pending_tp_orders <= 0:
                # 使用 GridStrategy 計算裝死模式價格 (統一回測/實盤邏輯)
                special_price = GridStrategy.calculate_dead_mode_price(
                    price, my_position, opposite_position, side
                )

                if side == 'long':
                    self.place_order(ccxt_symbol, 'sell', special_price, tp_qty, True, 'long')
                else:
                    self.place_order(ccxt_symbol, 'buy', special_price, tp_qty, True, 'short')
                logger.info(f"[MAX] {sym_config.symbol} {side}頭裝死止盈@{special_price:.4f}")
        else:
            # === 正常模式 ===
            if dead_mode_flag:
                # 狀態轉換: 離開裝死模式
                if side == 'long':
                    sym_state.long_dead_mode = False
                else:
                    sym_state.short_dead_mode = False
                logger.info(f"[MAX] {sym_config.symbol} {side}頭離開裝死模式")

            self.cancel_orders_for_side(ccxt_symbol, side)

            # 使用 GridStrategy 計算正常模式價格 (統一回測/實盤邏輯)
            tp_price, entry_price = GridStrategy.calculate_grid_prices(
                price, take_profit_spacing, grid_spacing, side
            )

            if side == 'long':
                if sym_state.long_position > 0:
                    self.place_order(ccxt_symbol, 'sell', tp_price, tp_qty, True, 'long')
                self.place_order(ccxt_symbol, 'buy', entry_price, base_qty, False, 'long')
            else:
                if sym_state.short_position > 0:
                    self.place_order(ccxt_symbol, 'buy', tp_price, tp_qty, True, 'short')
                self.place_order(ccxt_symbol, 'sell', entry_price, base_qty, False, 'short')

            logger.info(f"[MAX] {sym_config.symbol} {side}頭 止盈@{tp_price:.4f}({tp_qty:.1f}) "
                       f"補倉@{entry_price:.4f}({base_qty:.1f}) [TP:{take_profit_spacing*100:.2f}%/GS:{grid_spacing*100:.2f}%]")

    async def _handle_ticker(self, data: dict):
        symbol_raw = data.get('s', '')
        bid = float(data.get('b', 0))
        ask = float(data.get('a', 0))

        if not bid or not ask:
            return

        for sym_config in self.config.symbols.values():
            if sym_config.enabled and sym_config.ws_symbol.upper() == symbol_raw:
                ccxt_symbol = sym_config.ccxt_symbol
                state = self.state.symbols.get(ccxt_symbol)
                if state:
                    state.best_bid = bid
                    state.best_ask = ask
                    state.latest_price = (bid + ask) / 2

                    # === 領先指標: 更新價差數據 ===
                    self.leading_indicator.update_spread(ccxt_symbol, bid, ask)

                    await self.adjust_grid(ccxt_symbol)
                break

        if time.time() - self.last_sync_time > self.config.sync_interval:
            self.sync_all()
            self.last_sync_time = time.time()

    async def _handle_account_update(self, data: dict):
        """處理 ACCOUNT_UPDATE 事件"""
        try:
            account_data = data.get('a', {})

            # 分別更新 USDC 和 USDT 帳戶餘額
            balances = account_data.get('B', [])
            for bal in balances:
                asset = bal.get('a', '')
                if asset in ['USDC', 'USDT']:
                    wallet_balance = float(bal.get('wb', 0) or 0)
                    cross_wallet = float(bal.get('cw', 0) or 0)

                    acc = self.state.get_account(asset)
                    acc.wallet_balance = wallet_balance
                    acc.available_balance = cross_wallet

                    logger.info(f"[userData] {asset} 餘額更新: 錢包={wallet_balance:.2f}, 可用={cross_wallet:.2f}")

            for sym_state in self.state.symbols.values():
                sym_state.unrealized_pnl = 0

            positions = account_data.get('P', [])
            for pos in positions:
                symbol_raw = pos.get('s', '')
                position_amt = float(pos.get('pa', 0) or 0)
                unrealized_pnl = float(pos.get('up', 0) or 0)
                position_side = pos.get('ps', '')

                ccxt_symbol = None
                for cfg in self.config.symbols.values():
                    if cfg.symbol == symbol_raw:
                        ccxt_symbol = cfg.ccxt_symbol
                        break

                if ccxt_symbol and ccxt_symbol in self.state.symbols:
                    sym_state = self.state.symbols[ccxt_symbol]

                    if position_side == 'LONG':
                        sym_state.long_position = abs(position_amt)
                    elif position_side == 'SHORT':
                        sym_state.short_position = abs(position_amt)

                    sym_state.unrealized_pnl += unrealized_pnl

                    logger.info(f"[userData] {symbol_raw} {position_side}: "
                               f"持倉={position_amt:.2f}, 浮盈={unrealized_pnl:.2f}")

            # 更新各帳戶的未實現盈虧
            for currency in ['USDC', 'USDT']:
                acc = self.state.get_account(currency)
                acc.unrealized_pnl = sum(
                    s.unrealized_pnl for s in self.state.symbols.values()
                    if currency in s.symbol
                )

            # 更新總計
            self.state.update_totals()

        except Exception as e:
            logger.error(f"[userData] ACCOUNT_UPDATE 處理失敗: {e}")

    async def _handle_order_update(self, data: dict):
        """處理 ORDER_TRADE_UPDATE 事件"""
        try:
            order_data = data.get('o', {})
            symbol_raw = order_data.get('s', '')
            order_status = order_data.get('X', '')
            side = order_data.get('S', '')
            position_side = order_data.get('ps', '')
            realized_pnl = float(order_data.get('rp', 0) or 0)

            ccxt_symbol = None
            sym_config = None
            for cfg in self.config.symbols.values():
                if cfg.symbol == symbol_raw:
                    ccxt_symbol = cfg.ccxt_symbol
                    sym_config = cfg
                    break

            if not ccxt_symbol or ccxt_symbol not in self.state.symbols:
                return

            sym_state = self.state.symbols[ccxt_symbol]

            if order_status == 'FILLED':
                sym_state.total_trades += 1
                self.state.total_trades += 1

                # === 領先指標: 記錄成交 (用於計算 OFI) ===
                exec_price = float(order_data.get('p', 0) or order_data.get('ap', 0) or 0)
                exec_qty = float(order_data.get('q', 0) or 0)
                trade_side_for_ofi = 'buy' if side == 'BUY' else 'sell'
                if exec_price > 0 and exec_qty > 0:
                    self.leading_indicator.record_trade(ccxt_symbol, exec_price, exec_qty, trade_side_for_ofi)

                if realized_pnl != 0:
                    sym_state.total_profit += realized_pnl
                    self.state.total_profit += realized_pnl
                    pnl_sign = "+" if realized_pnl > 0 else ""
                    logger.info(f"[userData] {symbol_raw} 成交! {side} {position_side}, "
                               f"盈虧: {pnl_sign}{realized_pnl:.4f}")

                    # === Bandit 記錄交易結果 ===
                    trade_side = 'long' if position_side == 'LONG' else 'short'
                    self.bandit_optimizer.record_trade(realized_pnl, trade_side)

                    # === DGT 累積利潤 ===
                    self.dgt_manager.accumulated_profits[ccxt_symbol] = \
                        self.dgt_manager.accumulated_profits.get(ccxt_symbol, 0) + realized_pnl
                else:
                    logger.info(f"[userData] {symbol_raw} 開倉成交: {side} {position_side}")

                if position_side == 'LONG':
                    if side == 'BUY':
                        sym_state.buy_long_orders = 0
                    else:
                        sym_state.sell_long_orders = 0
                elif position_side == 'SHORT':
                    if side == 'SELL':
                        sym_state.sell_short_orders = 0
                    else:
                        sym_state.buy_short_orders = 0

                await self.adjust_grid(ccxt_symbol)

            elif order_status == 'CANCELED':
                logger.info(f"[userData] {symbol_raw} 訂單取消: {side} {position_side}")

        except Exception as e:
            logger.error(f"[userData] ORDER_TRADE_UPDATE 處理失敗: {e}")

    async def _websocket_loop(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.config.websocket_url, ssl=ssl_context) as ws:
                    self.state.connected = True

                    streams = []
                    for cfg in self.config.symbols.values():
                        if cfg.enabled:
                            streams.append(f"{cfg.ws_symbol}@bookTicker")

                    if streams:
                        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))

                    if self.listen_key:
                        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": [self.listen_key], "id": 2}))
                        logger.info("[WebSocket] 已訂閱 userData stream")

                    while not self._stop_event.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)

                            event_type = data.get('e', '')

                            if event_type == 'bookTicker':
                                await self._handle_ticker(data)
                            elif event_type == 'ACCOUNT_UPDATE':
                                await self._handle_account_update(data)
                            elif event_type == 'ORDER_TRADE_UPDATE':
                                await self._handle_order_update(data)

                        except asyncio.TimeoutError:
                            await ws.ping()
            except Exception as e:
                self.state.connected = False
                if not self._stop_event.is_set():
                    logger.error(f"WebSocket 錯誤: {e}")
                    await asyncio.sleep(5)

    async def _keep_alive_loop(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(1800)
                if not self._stop_event.is_set():
                    self.exchange.fapiPrivatePutListenKey()
                    self.listen_key = self._get_listen_key()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"更新 listenKey 失敗: {e}")

    async def run(self):
        try:
            self._init_exchange()
            self._check_hedge_mode()
            self.listen_key = self._get_listen_key()

            # 提前設置 running，讓主線程知道初始化進行中
            self.state.running = True
            self.state.start_time = datetime.now()

            self.sync_all()
        except Exception as e:
            logger.error(f"[MAX] 初始化失敗: {e}")
            self.state.running = False
            return

        self.tasks = [
            asyncio.create_task(self._websocket_loop()),
            asyncio.create_task(self._keep_alive_loop())
        ]

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            await self.stop()

    async def stop(self):
        self._stop_event.set()
        self.state.running = False

        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              終端 UI (MAX 版本)                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TerminalUI:
    def __init__(self, config: GlobalConfig, state: GlobalState, bot: 'MaxGridBot' = None):
        self.config = config
        self.state = state
        self.bot = bot

    def create_header(self) -> Panel:
        status = "[green]● 運行中[/]" if self.state.running else "[red]● 已停止[/]"
        ws_status = "[green]WS[/]" if self.state.connected else "[yellow]WS斷開[/]"

        if self.state.start_time:
            duration = datetime.now() - self.state.start_time
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            runtime = "--:--:--"

        enabled_count = len([s for s in self.config.symbols.values() if s.enabled])

        header = Text()
        header.append("AS 網格交易系統 ", style="bold cyan")
        header.append("MAX ", style="bold yellow")
        header.append(f"│ {status} │ {ws_status} ", style="")
        header.append(f"│ {enabled_count} 交易對 ", style="dim")
        header.append(f"│ {runtime}", style="dim")

        return Panel(header, box=box.DOUBLE_EDGE, style="cyan")

    def create_account_panel(self) -> Panel:
        table = Table(box=None, show_header=False, expand=True, padding=(0, 1))
        table.add_column("", style="dim", width=8)
        table.add_column("USDC", justify="right", width=12)
        table.add_column("USDT", justify="right", width=12)

        # 標題行
        table.add_row("", "[cyan bold]USDC[/]", "[yellow bold]USDT[/]")

        # 分帳戶顯示
        usdc = self.state.get_account("USDC")
        usdt = self.state.get_account("USDT")

        # 權益
        table.add_row(
            "權益",
            f"[white]{usdc.equity:.2f}[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[white]{usdt.equity:.2f}[/]" if usdt.equity > 0 else "[dim]--[/]"
        )

        # 可用
        table.add_row(
            "可用",
            f"{usdc.available_balance:.2f}" if usdc.available_balance > 0 else "[dim]--[/]",
            f"{usdt.available_balance:.2f}" if usdt.available_balance > 0 else "[dim]--[/]"
        )

        # 保證金率
        def margin_style(ratio):
            if ratio < 0.3:
                return "green"
            elif ratio < 0.6:
                return "yellow"
            else:
                return "red"

        table.add_row(
            "保證金",
            f"[{margin_style(usdc.margin_ratio)}]{usdc.margin_ratio*100:.1f}%[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[{margin_style(usdt.margin_ratio)}]{usdt.margin_ratio*100:.1f}%[/]" if usdt.equity > 0 else "[dim]--[/]"
        )

        # 浮盈
        def pnl_style(pnl):
            return "green" if pnl >= 0 else "red"

        table.add_row(
            "浮盈",
            f"[{pnl_style(usdc.unrealized_pnl)}]{'+' if usdc.unrealized_pnl >= 0 else ''}{usdc.unrealized_pnl:.2f}[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[{pnl_style(usdt.unrealized_pnl)}]{'+' if usdt.unrealized_pnl >= 0 else ''}{usdt.unrealized_pnl:.2f}[/]" if usdt.equity > 0 else "[dim]--[/]"
        )

        # 總計
        table.add_row("", "", "")
        pnl_color = "green" if self.state.total_unrealized_pnl >= 0 else "red"
        pnl_sign = "+" if self.state.total_unrealized_pnl >= 0 else ""

        table.add_row("[bold]總計[/]", f"[bold white]{self.state.total_equity:.2f}[/]", f"[{pnl_color}]{pnl_sign}{self.state.total_unrealized_pnl:.2f}[/]")

        return Panel(table, title="[bold]帳戶 (USDC | USDT)[/]", box=box.ROUNDED)

    def create_symbols_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("交易對", style="cyan")
        table.add_column("價格", justify="right")
        table.add_column("多", justify="right")
        table.add_column("空", justify="right")
        table.add_column("浮盈", justify="right")
        table.add_column("狀態", justify="center")
        table.add_column("TP/GS", justify="right")

        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue

            sym_state = self.state.symbols.get(sym_config.ccxt_symbol)
            if not sym_state:
                continue

            price_str = f"{sym_state.latest_price:.4f}" if sym_state.latest_price else "--"
            long_style = "green" if sym_state.long_position > 0 else "dim"
            short_style = "red" if sym_state.short_position > 0 else "dim"
            pnl = sym_state.unrealized_pnl
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""

            # 狀態顯示 (裝死/加倍/正常)
            status_parts = []
            # 多頭狀態
            if sym_state.long_position > sym_config.position_threshold:
                status_parts.append("[red bold]多裝死[/]")
            elif sym_state.long_position > sym_config.position_limit:
                status_parts.append("[yellow]多×2[/]")
            # 空頭狀態
            if sym_state.short_position > sym_config.position_threshold:
                status_parts.append("[red bold]空裝死[/]")
            elif sym_state.short_position > sym_config.position_limit:
                status_parts.append("[yellow]空×2[/]")

            if not status_parts:
                # 顯示距離閾值的進度
                long_pct = (sym_state.long_position / sym_config.position_limit * 100) if sym_config.position_limit > 0 else 0
                short_pct = (sym_state.short_position / sym_config.position_limit * 100) if sym_config.position_limit > 0 else 0
                if long_pct > 50 or short_pct > 50:
                    status_str = f"[dim]{max(long_pct, short_pct):.0f}%[/]"
                else:
                    status_str = "[dim green]正常[/]"
            else:
                status_str = " ".join(status_parts)

            # 動態間距顯示
            if sym_state.dynamic_take_profit > 0:
                spacing_str = f"{sym_state.dynamic_take_profit*100:.2f}/{sym_state.dynamic_grid_spacing*100:.2f}"
            else:
                spacing_str = f"{sym_config.take_profit_spacing*100:.2f}/{sym_config.grid_spacing*100:.2f}"

            table.add_row(
                f"{sym_config.coin_name}",
                price_str,
                f"[{long_style}]{sym_state.long_position:.1f}[/]",
                f"[{short_style}]{sym_state.short_position:.1f}[/]",
                f"[{pnl_style}]{pnl_sign}{pnl:.2f}[/]",
                status_str,
                f"[dim]{spacing_str}[/]"
            )

        return Panel(table, title="[bold]交易對狀態[/]", box=box.ROUNDED)

    def create_max_panel(self) -> Panel:
        """MAX 增強功能狀態面板 (AS 高頻網格版)"""
        max_cfg = self.config.max_enhancement
        bandit_cfg = self.config.bandit
        leading_cfg = self.config.leading_indicator

        table = Table(box=None, show_header=False, expand=True)
        table.add_column("", style="dim")
        table.add_column("", justify="right")

        # Bandit 學習狀態 (AS 網格核心)
        if bandit_cfg.enabled and hasattr(self, 'bot') and self.bot:
            bandit = self.bot.bandit_optimizer
            params = bandit.get_current_params()
            # 顯示學習次數和當前間距
            table.add_row("學習次數", f"[green]#{bandit.total_pulls}[/]")
            table.add_row("當前間距", f"[cyan]{params.grid_spacing*100:.1f}%/{params.take_profit_spacing*100:.1f}%[/]")
            table.add_row("γ係數", f"{params.gamma:.2f}")
        else:
            table.add_row("Bandit", "[dim]OFF[/]")

        # 領先指標狀態
        if leading_cfg.enabled and hasattr(self, 'bot') and self.bot:
            # 取第一個交易對的領先指標數據作為示例
            for sym_state in self.state.symbols.values():
                ofi = sym_state.leading_ofi
                vol = sym_state.leading_volume_ratio
                spread = sym_state.leading_spread_ratio
                signals = sym_state.leading_signals

                # OFI 顏色
                ofi_style = "green" if ofi > 0.3 else "red" if ofi < -0.3 else "dim"
                ofi_str = f"[{ofi_style}]{ofi:+.2f}[/]"

                # Volume 顏色
                vol_style = "yellow" if vol > 2.0 else "dim"
                vol_str = f"[{vol_style}]{vol:.1f}x[/]"

                # Spread 顏色
                spread_style = "yellow" if spread > 1.5 else "dim"
                spread_str = f"[{spread_style}]{spread:.1f}x[/]"

                table.add_row("OFI/Vol/Spd", f"{ofi_str} {vol_str} {spread_str}")

                # 活躍信號
                if signals:
                    sig_str = ",".join([s.replace("_", "").replace("PRESSURE", "")[:6] for s in signals[:2]])
                    table.add_row("信號", f"[yellow]{sig_str}[/]")
                break
        else:
            table.add_row("領先指標", "[dim]OFF[/]")

        # 增強模式狀態
        if max_cfg.all_enhancements_enabled:
            table.add_row("增強模式", "[green]ON[/]")
        else:
            table.add_row("增強模式", "[dim]OFF[/]")

        return Panel(table, title="[bold yellow]AS 學習[/]", box=box.ROUNDED)

    def create_help_panel(self) -> Panel:
        help_text = Text()
        help_text.append("[Ctrl+C]", style="bold cyan")
        help_text.append(" 退出", style="dim")
        return Panel(help_text, box=box.ROUNDED, style="dim")

    def create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="left", size=38),
            Layout(name="symbols")
        )
        layout["left"].split_column(
            Layout(name="account"),
            Layout(name="max", size=10)
        )
        layout["header"].update(self.create_header())
        layout["account"].update(self.create_account_panel())
        layout["max"].update(self.create_max_panel())
        layout["symbols"].update(self.create_symbols_panel())
        layout["footer"].update(self.create_help_panel())
        return layout


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              主菜單                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MainMenu:
    def __init__(self):
        self.config = GlobalConfig.load()
        self.backtest_manager = BacktestManager()

        # 背景交易相關
        self.bot: Optional[MaxGridBot] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_loop: Optional[asyncio.AbstractEventLoop] = None
        self._trading_active = False

    def show_banner(self):
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]AS 網格交易系統[/] [bold yellow]MAX[/]\n"
            "[dim]Funding Rate · GLFT · 動態網格[/]",
            border_style="yellow"
        ))
        console.print()

    def main_menu(self):
        while True:
            self.show_banner()

            # 顯示交易狀態
            if self._trading_active and self.bot:
                console.print("[bold green]● 交易運行中[/]", end="  ")
                if self.bot.state.start_time:
                    duration = datetime.now() - self.bot.state.start_time
                    hours, remainder = divmod(int(duration.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    console.print(f"[dim]運行時間: {hours:02d}:{minutes:02d}:{seconds:02d}[/]", end="  ")
                console.print(f"[dim]浮盈: {self.bot.state.total_unrealized_pnl:+.2f}[/]\n")

            if self.config.symbols:
                enabled = [s for s in self.config.symbols.values() if s.enabled]
                console.print(f"[dim]已配置 {len(self.config.symbols)} 個交易對，{len(enabled)} 個啟用[/]\n")

            console.print("[bold]請選擇操作:[/]\n")

            # 根據交易狀態顯示不同選項
            if self._trading_active:
                console.print("  [cyan]1[/] 查看交易面板")
                console.print("  [cyan]s[/] [red]停止交易[/]")
            else:
                console.print("  [cyan]1[/] 開始交易")

            console.print("  [cyan]2[/] 管理交易對")
            console.print("  [cyan]3[/] 回測/優化")
            console.print("  [cyan]4[/] MAX 增強設定")
            console.print("  [cyan]5[/] 學習模組 (Bandit/DGT)")
            console.print("  [cyan]6[/] 風控設定")
            console.print("  [cyan]7[/] API 設定")
            console.print("  [cyan]0[/] 退出")
            console.print()

            valid_choices = ["0", "1", "2", "3", "4", "5", "6", "7"]
            if self._trading_active:
                valid_choices.append("s")

            choice = Prompt.ask("選擇", choices=valid_choices, default="1")

            if choice == "0":
                if self._trading_active:
                    if Confirm.ask("[yellow]交易運行中，確定要退出嗎？[/]"):
                        self.stop_trading()
                        break
                else:
                    break
            elif choice == "1":
                if self._trading_active:
                    self.view_trading_panel()
                else:
                    self.start_trading()
            elif choice == "s" and self._trading_active:
                self.stop_trading()
            elif choice == "2":
                self.manage_symbols()
            elif choice == "3":
                self.quick_backtest()
            elif choice == "4":
                self.setup_max_enhancement()
            elif choice == "5":
                self.setup_learning()
            elif choice == "6":
                self.setup_risk()
            elif choice == "7":
                self.setup_api()

    def quick_backtest(self):
        """快速回測 - 直接輸入交易對，自動下載+優化"""
        self.show_banner()
        console.print("[bold]回測/優化[/]\n")
        console.print("[dim]直接輸入交易對符號，如: XRPUSDC, BTCUSDT[/]\n")

        symbol_input = Prompt.ask("交易對").strip()
        raw, ccxt_sym, coin, quote = normalize_symbol(symbol_input)

        if not raw:
            console.print(f"[red]無法識別交易對: {symbol_input}[/]")
            console.print("[dim]支援格式: XRPUSDC, BTCUSDT, ETH/USDC 等[/]")
            Prompt.ask("按 Enter 繼續")
            return

        console.print(f"\n[green]識別為: {coin}/{quote} ({raw})[/]\n")

        # 檢查現有數據
        available_dates = self.backtest_manager.get_available_dates(raw)

        if available_dates:
            console.print(f"[dim]已有數據: {available_dates[0]} 至 {available_dates[-1]}[/]\n")

        # 日期選擇
        today = datetime.now()
        console.print("[bold]選擇回測時間範圍:[/]\n")
        console.print("  [cyan]1[/] 最近 7 天  (1W)")
        console.print("  [cyan]2[/] 最近 14 天 (2W)")
        console.print("  [cyan]3[/] 最近 30 天 (1M)")
        console.print("  [cyan]4[/] 最近 90 天 (3M)")
        console.print("  [cyan]5[/] 最近 180 天 (6M)")
        console.print("  [cyan]6[/] 最近 365 天 (1Y)")
        console.print("  [cyan]7[/] 自定義日期範圍")
        console.print()

        date_choice = Prompt.ask("選擇", choices=["1", "2", "3", "4", "5", "6", "7"], default="3")

        # 計算日期範圍
        date_ranges = {
            "1": 7,
            "2": 14,
            "3": 30,
            "4": 90,
            "5": 180,
            "6": 365,
        }

        if date_choice == "7":
            # 自定義日期
            default_start = available_dates[0] if available_dates else (today - timedelta(days=30)).strftime("%Y-%m-%d")
            default_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = Prompt.ask("開始日期 (YYYY-MM-DD)", default=default_start)
            end_date = Prompt.ask("結束日期 (YYYY-MM-DD)", default=default_end)
        else:
            days = date_ranges[date_choice]
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
            console.print(f"\n[dim]時間範圍: {start_date} 至 {end_date}[/]")

        # 檢查是否需要下載
        need_download = False
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            current = start_dt
            while current <= end_dt:
                date_str = current.strftime("%Y-%m-%d")
                if date_str not in available_dates:
                    need_download = True
                    break
                current += timedelta(days=1)
        except ValueError:
            console.print("[red]日期格式錯誤[/]")
            Prompt.ask("按 Enter 繼續")
            return

        if need_download:
            console.print("\n[yellow]部分數據缺失，開始下載...[/]\n")
            if not self.backtest_manager.download_data(raw, ccxt_sym, start_date, end_date):
                console.print("[red]下載失敗[/]")
                Prompt.ask("按 Enter 繼續")
                return

        # 載入數據
        console.print("\n載入數據...")
        df = self.backtest_manager.load_data(raw, start_date, end_date)

        if df is None or df.empty:
            console.print("[red]載入數據失敗[/]")
            Prompt.ask("按 Enter 繼續")
            return

        console.print(f"[green]載入 {len(df):,} 條 K 線[/]\n")

        # 選擇模式
        console.print("  [cyan]1[/] 執行回測 (使用當前/默認參數)")
        console.print("  [cyan]2[/] 參數優化 (搜索最佳參數)")
        console.print()

        mode = Prompt.ask("選擇", choices=["1", "2"], default="2")

        # 獲取或創建配置
        if raw in self.config.symbols:
            sym_config = self.config.symbols[raw]
        else:
            sym_config = SymbolConfig(symbol=raw, ccxt_symbol=ccxt_sym)

        if mode == "1":
            # 單次回測
            console.print("\n執行回測...\n")
            result = self.backtest_manager.run_backtest(sym_config, df)
            self._show_backtest_result(result)

        else:
            # 參數優化
            console.print("\n執行參數優化...\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("優化中...", total=100)

                def update_progress(current, total):
                    progress.update(task, completed=current * 100 // total)

                results = self.backtest_manager.optimize_params(sym_config, df, update_progress)

            # 顯示結果
            console.print("\n" + "="*60)
            console.print("[bold cyan]優化結果 (Top 5)[/]")
            console.print("="*60 + "\n")

            table = Table(box=box.ROUNDED)
            table.add_column("#", style="dim")
            table.add_column("止盈", justify="right")
            table.add_column("補倉", justify="right")
            table.add_column("收益率", justify="right")
            table.add_column("回撤", justify="right")
            table.add_column("交易數", justify="right")
            table.add_column("勝率", justify="right")

            for i, r in enumerate(results[:5], 1):
                return_color = "green" if r["return_pct"] >= 0 else "red"
                table.add_row(
                    str(i),
                    f"{r['take_profit_spacing']*100:.2f}%",
                    f"{r['grid_spacing']*100:.2f}%",
                    f"[{return_color}]{r['return_pct']*100:.2f}%[/]",
                    f"{r['max_drawdown']*100:.2f}%",
                    str(r['trades_count']),
                    f"{r['win_rate']*100:.1f}%"
                )

            console.print(table)

            # 詢問是否應用
            if results:
                console.print()
                console.print(f"[dim]當前參數: 止盈 {sym_config.take_profit_spacing*100:.2f}%, 補倉 {sym_config.grid_spacing*100:.2f}%[/]")
                console.print(f"[bold]最佳參數: 止盈 {results[0]['take_profit_spacing']*100:.2f}%, 補倉 {results[0]['grid_spacing']*100:.2f}%[/]")
                console.print()

                if Confirm.ask("是否應用最佳參數?"):
                    best = results[0]

                    # 更新或創建配置
                    if raw not in self.config.symbols:
                        self.config.symbols[raw] = sym_config

                    self.config.symbols[raw].take_profit_spacing = best["take_profit_spacing"]
                    self.config.symbols[raw].grid_spacing = best["grid_spacing"]
                    self.config.save()

                    console.print(f"\n[green]已應用並保存: 止盈 {best['take_profit_spacing']*100:.2f}%, 補倉 {best['grid_spacing']*100:.2f}%[/]")

        Prompt.ask("\n按 Enter 繼續")

    def _show_backtest_result(self, result: dict):
        """顯示回測結果"""
        console.print("="*50)
        console.print("[bold cyan]回測結果[/]")
        console.print("="*50 + "\n")

        return_color = "green" if result["return_pct"] >= 0 else "red"

        table = Table(box=box.ROUNDED)
        table.add_column("指標", style="dim")
        table.add_column("值", justify="right")

        table.add_row("最終淨值", f"${result['final_equity']:.2f}")
        table.add_row("收益率", f"[{return_color}]{result['return_pct']*100:.2f}%[/]")
        table.add_row("最大回撤", f"[red]{result['max_drawdown']*100:.2f}%[/]")
        table.add_row("交易次數", str(result['trades_count']))
        table.add_row("勝率", f"{result['win_rate']*100:.1f}%")
        pf = result['profit_factor']
        table.add_row("盈虧比", f"{pf:.2f}" if pf != float('inf') else "∞")
        table.add_row("已實現盈虧", f"${result['realized_pnl']:.2f}")
        table.add_row("未實現盈虧", f"${result['unrealized_pnl']:.2f}")

        console.print(table)

    def setup_max_enhancement(self):
        """MAX 增強功能設定"""
        self.show_banner()
        console.print("[bold yellow]MAX 增強功能設定[/]\n")

        max_cfg = self.config.max_enhancement

        # 顯示總開關狀態
        if max_cfg.all_enhancements_enabled:
            mode_status = "[bold green]增強模式[/] (學術模型啟用)"
        else:
            mode_status = "[bold cyan]純淨模式[/] (與 Pro 版相同)"
        console.print(f"[bold]當前模式:[/] {mode_status}\n")

        # 顯示當前設定
        console.print("[bold]1. Funding Rate 偏向[/]")
        fr_active = max_cfg.is_feature_enabled('funding_rate')
        fr_status = "[green]啟用[/]" if fr_active else "[dim]停用[/]"
        console.print(f"   狀態: {fr_status}")
        console.print(f"   閾值: {max_cfg.funding_rate_threshold*100:.3f}% (超過才調整)")
        console.print(f"   偏向強度: {max_cfg.funding_rate_position_bias*100:.0f}%")
        console.print()

        console.print("[bold]2. GLFT 庫存控制[/]")
        glft_active = max_cfg.is_feature_enabled('glft')
        glft_status = "[green]啟用[/]" if glft_active else "[dim]停用[/]"
        console.print(f"   狀態: {glft_status}")
        console.print(f"   γ (風險厭惡): {max_cfg.gamma}")
        console.print(f"   目標庫存比: {max_cfg.inventory_target}")
        console.print()

        console.print("[bold]3. 動態網格範圍[/]")
        dg_active = max_cfg.is_feature_enabled('dynamic_grid')
        dg_status = "[green]啟用[/]" if dg_active else "[dim]停用[/]"
        console.print(f"   狀態: {dg_status}")
        console.print(f"   ATR 週期: {max_cfg.atr_period}")
        console.print(f"   ATR 乘數: {max_cfg.atr_multiplier}")
        console.print(f"   間距範圍: {max_cfg.min_spacing*100:.2f}% ~ {max_cfg.max_spacing*100:.2f}%")
        console.print()

        if not Confirm.ask("是否修改設定?"):
            return

        # 總開關
        console.print("\n[bold yellow]── 模式選擇 ──[/]")
        console.print("[dim]純淨模式: 與 Pro 版完全相同，固定間距和數量[/]")
        console.print("[dim]增強模式: 啟用學術模型，動態調整間距和數量[/]")
        max_cfg.all_enhancements_enabled = Confirm.ask(
            "啟用增強模式?",
            default=max_cfg.all_enhancements_enabled
        )

        if not max_cfg.all_enhancements_enabled:
            self.config.save()
            console.print("\n[cyan]已切換到純淨模式，與 Pro 版行為相同[/]")
            Prompt.ask("按 Enter 繼續")
            return

        # Funding Rate 設定
        console.print("\n[bold cyan]── Funding Rate 偏向 ──[/]")
        max_cfg.funding_rate_enabled = Confirm.ask("啟用 Funding Rate 偏向?", default=max_cfg.funding_rate_enabled)
        if max_cfg.funding_rate_enabled:
            max_cfg.funding_rate_threshold = FloatPrompt.ask(
                f"閾值 (%) [當前: {max_cfg.funding_rate_threshold*100:.3f}]",
                default=max_cfg.funding_rate_threshold * 100
            ) / 100
            max_cfg.funding_rate_position_bias = FloatPrompt.ask(
                f"偏向強度 (%) [當前: {max_cfg.funding_rate_position_bias*100:.0f}]",
                default=max_cfg.funding_rate_position_bias * 100
            ) / 100

        # GLFT 設定
        console.print("\n[bold cyan]── GLFT 庫存控制 ──[/]")
        max_cfg.glft_enabled = Confirm.ask("啟用 GLFT 庫存控制?", default=max_cfg.glft_enabled)
        if max_cfg.glft_enabled:
            max_cfg.gamma = FloatPrompt.ask(
                f"γ 風險厭惡係數 (0.01-1.0) [當前: {max_cfg.gamma}]",
                default=max_cfg.gamma
            )

        # 動態網格設定
        console.print("\n[bold cyan]── 動態網格範圍 ──[/]")
        max_cfg.dynamic_grid_enabled = Confirm.ask("啟用動態網格?", default=max_cfg.dynamic_grid_enabled)
        if max_cfg.dynamic_grid_enabled:
            max_cfg.atr_period = IntPrompt.ask(
                f"ATR 週期 [當前: {max_cfg.atr_period}]",
                default=max_cfg.atr_period
            )
            max_cfg.atr_multiplier = FloatPrompt.ask(
                f"ATR 乘數 [當前: {max_cfg.atr_multiplier}]",
                default=max_cfg.atr_multiplier
            )
            max_cfg.min_spacing = FloatPrompt.ask(
                f"最小間距 (%) [當前: {max_cfg.min_spacing*100:.2f}]",
                default=max_cfg.min_spacing * 100
            ) / 100
            max_cfg.max_spacing = FloatPrompt.ask(
                f"最大間距 (%) [當前: {max_cfg.max_spacing*100:.2f}]",
                default=max_cfg.max_spacing * 100
            ) / 100

        self.config.save()
        console.print("[green]MAX 增強設定已保存[/]")

        # 如果交易運行中，即時更新
        if self._trading_active and self.bot:
            self.bot.config = self.config
            console.print("[cyan]✓ 配置已即時套用[/]")

        Prompt.ask("按 Enter 繼續")

    def setup_api(self):
        self.show_banner()
        console.print("[bold]API 設定[/]\n")

        if self.config.api_key:
            console.print(f"[dim]當前 API Key: {self.config.api_key[:8]}...{self.config.api_key[-4:]}[/]")
            if not Confirm.ask("是否重新設定?"):
                return

        self.config.api_key = Prompt.ask("API Key")
        self.config.api_secret = Prompt.ask("API Secret")
        self.config.save()

        console.print("[green]API 設定完成[/]")
        Prompt.ask("按 Enter 繼續")

    def setup_learning(self):
        """學習模組設定 (Bandit + 領先指標 + DGT)"""
        self.show_banner()
        console.print("[bold yellow]學習模組設定[/]\n")

        bandit = self.config.bandit
        leading = self.config.leading_indicator
        dgt = self.config.dgt

        # === Bandit 設定 ===
        console.print("[bold]1. UCB Bandit 參數優化器[/]")
        console.print("[dim]   基於 TradeBot 論文，自動學習最佳參數組合[/]")
        bandit_status = "[green]啟用[/]" if bandit.enabled else "[red]停用[/]"
        console.print(f"   狀態: {bandit_status}")
        console.print(f"   滑動窗口: {bandit.window_size} 筆交易")
        console.print(f"   探索係數: {bandit.exploration_factor}")
        console.print(f"   更新間隔: 每 {bandit.update_interval} 筆交易")
        console.print()

        # === 領先指標設定 ===
        console.print("[bold]2. 領先指標系統 (取代滯後 ATR)[/]")
        console.print("[dim]   OFI (訂單流) + Volume (成交量) + Spread (價差)[/]")
        leading_status = "[green]啟用[/]" if leading.enabled else "[red]停用[/]"
        console.print(f"   狀態: {leading_status}")
        console.print(f"   OFI 閾值: ±{leading.ofi_threshold}")
        console.print(f"   放量倍數: {leading.volume_surge_threshold}x")
        console.print(f"   價差倍數: {leading.spread_surge_threshold}x")
        console.print()

        # === DGT 設定 ===
        console.print("[bold]3. DGT 動態邊界重置[/]")
        console.print("[dim]   基於 arXiv:2506.11921，價格突破邊界時自動重置網格[/]")
        dgt_status = "[green]啟用[/]" if dgt.enabled else "[red]停用[/]"
        console.print(f"   狀態: {dgt_status}")
        console.print(f"   邊界緩衝: {dgt.boundary_buffer*100:.1f}%")
        console.print(f"   利潤再投資: {dgt.profit_reinvest_ratio*100:.0f}%")
        console.print()

        if not Confirm.ask("是否修改設定?"):
            return

        # Bandit 設定
        console.print("\n[bold cyan]── UCB Bandit 設定 ──[/]")
        bandit.enabled = Confirm.ask("啟用 Bandit 參數學習?", default=bandit.enabled)
        if bandit.enabled:
            bandit.window_size = IntPrompt.ask(
                f"滑動窗口大小 [當前: {bandit.window_size}]",
                default=bandit.window_size
            )
            bandit.exploration_factor = FloatPrompt.ask(
                f"探索係數 (1.0-3.0) [當前: {bandit.exploration_factor}]",
                default=bandit.exploration_factor
            )
            bandit.update_interval = IntPrompt.ask(
                f"更新間隔 (筆交易) [當前: {bandit.update_interval}]",
                default=bandit.update_interval
            )

        # 領先指標設定
        console.print("\n[bold cyan]── 領先指標設定 ──[/]")
        console.print("[dim]領先指標預測波動，優先於滯後的 ATR[/]")
        leading.enabled = Confirm.ask("啟用領先指標?", default=leading.enabled)
        if leading.enabled:
            leading.ofi_enabled = Confirm.ask("  啟用 OFI (訂單流失衡)?", default=leading.ofi_enabled)
            leading.volume_enabled = Confirm.ask("  啟用成交量分析?", default=leading.volume_enabled)
            leading.spread_enabled = Confirm.ask("  啟用價差分析?", default=leading.spread_enabled)

            leading.ofi_threshold = FloatPrompt.ask(
                f"  OFI 觸發閾值 (0-1) [當前: {leading.ofi_threshold}]",
                default=leading.ofi_threshold
            )
            leading.volume_surge_threshold = FloatPrompt.ask(
                f"  放量倍數閾值 [當前: {leading.volume_surge_threshold}]",
                default=leading.volume_surge_threshold
            )
            leading.spread_surge_threshold = FloatPrompt.ask(
                f"  價差倍數閾值 [當前: {leading.spread_surge_threshold}]",
                default=leading.spread_surge_threshold
            )

        # DGT 設定
        console.print("\n[bold cyan]── DGT 動態邊界 ──[/]")
        dgt.enabled = Confirm.ask("啟用 DGT 邊界重置?", default=dgt.enabled)
        if dgt.enabled:
            dgt.boundary_buffer = FloatPrompt.ask(
                f"邊界緩衝 (%) [當前: {dgt.boundary_buffer*100:.1f}]",
                default=dgt.boundary_buffer * 100
            ) / 100
            dgt.profit_reinvest_ratio = FloatPrompt.ask(
                f"利潤再投資比例 (%) [當前: {dgt.profit_reinvest_ratio*100:.0f}]",
                default=dgt.profit_reinvest_ratio * 100
            ) / 100

        self.config.save()
        console.print("\n[green]學習模組設定已保存[/]")

        # 如果交易運行中，即時更新
        if self._trading_active and self.bot:
            self.bot.config = self.config
            console.print("[cyan]✓ 配置已即時套用[/]")

        Prompt.ask("按 Enter 繼續")

    def setup_risk(self):
        """風控設定"""
        self.show_banner()
        console.print("[bold]風控設定 - 保證金追蹤止盈[/]\n")

        risk = self.config.risk

        console.print("[dim]當前設定:[/]")
        status = "[green]啟用[/]" if risk.enabled else "[red]停用[/]"
        console.print(f"  狀態: {status}")
        console.print(f"  保證金閾值: {risk.margin_threshold*100:.0f}%")
        console.print(f"  啟動追蹤: 浮盈 >= {risk.trailing_start_profit:.1f}U")
        console.print(f"  回撤觸發: max({risk.trailing_min_drawdown:.1f}U, 最高浮盈 × {risk.trailing_drawdown_pct*100:.0f}%)")
        console.print()

        if Confirm.ask("是否修改設定?"):
            risk.enabled = Confirm.ask("啟用追蹤止盈?", default=risk.enabled)

            if risk.enabled:
                risk.margin_threshold = FloatPrompt.ask(
                    f"保證金閾值 (%) [當前: {risk.margin_threshold*100:.0f}]",
                    default=risk.margin_threshold * 100
                ) / 100

                risk.trailing_start_profit = FloatPrompt.ask(
                    f"啟動追蹤閾值 (U) [當前: {risk.trailing_start_profit:.1f}]",
                    default=risk.trailing_start_profit
                )

                risk.trailing_drawdown_pct = FloatPrompt.ask(
                    f"回撤比例 (%) [當前: {risk.trailing_drawdown_pct*100:.0f}]",
                    default=risk.trailing_drawdown_pct * 100
                ) / 100

                risk.trailing_min_drawdown = FloatPrompt.ask(
                    f"最小回撤 (U) [當前: {risk.trailing_min_drawdown:.1f}]",
                    default=risk.trailing_min_drawdown
                )

            self.config.save()
            console.print("[green]風控設定已保存[/]")

            # 如果交易運行中，即時更新
            if self._trading_active and self.bot:
                self.bot.config = self.config
                console.print("[cyan]✓ 配置已即時套用[/]")

        Prompt.ask("按 Enter 繼續")

    def manage_symbols(self):
        while True:
            self.show_banner()
            console.print("[bold]交易對管理[/]\n")

            # 如果交易運行中，顯示提示
            if self._trading_active:
                console.print("[dim yellow]● 交易運行中 - 修改參數會即時套用[/]\n")

            if self.config.symbols:
                table = Table(box=box.ROUNDED)
                table.add_column("#", style="dim")
                table.add_column("交易對", style="cyan")
                table.add_column("狀態")
                table.add_column("止盈", justify="right")
                table.add_column("補倉", justify="right")
                table.add_column("數量", justify="right")
                table.add_column("槓桿", justify="right")
                table.add_column("加倍", justify="right", style="yellow")
                table.add_column("裝死", justify="right", style="red")

                for i, cfg in enumerate(self.config.symbols.values(), 1):
                    status = "[green]啟用[/]" if cfg.enabled else "[dim]停用[/]"
                    table.add_row(
                        str(i),
                        cfg.symbol,
                        status,
                        f"{cfg.take_profit_spacing*100:.2f}%",
                        f"{cfg.grid_spacing*100:.2f}%",
                        str(cfg.initial_quantity),
                        f"{cfg.leverage}x",
                        f"×{cfg.limit_multiplier:.0f} ({cfg.position_limit:.0f})",
                        f"×{cfg.threshold_multiplier:.0f} ({cfg.position_threshold:.0f})"
                    )

                console.print(table)
                console.print()

            console.print("  [cyan]a[/] 新增交易對")
            console.print("  [cyan]e[/] 編輯交易對")
            console.print("  [cyan]d[/] 刪除交易對")
            console.print("  [cyan]t[/] 切換啟用/停用")
            console.print("  [cyan]0[/] 返回")
            console.print()

            choice = Prompt.ask("選擇", choices=["0", "a", "e", "d", "t"], default="0")

            if choice == "0":
                break
            elif choice == "a":
                self.add_symbol()
            elif choice == "e":
                self.edit_symbol()
            elif choice == "d":
                self.delete_symbol()
            elif choice == "t":
                self.toggle_symbol()

    def add_symbol(self):
        self.show_banner()
        console.print("[bold]新增交易對[/]\n")

        symbol_input = Prompt.ask("輸入交易對 (如 XRPUSDC)")
        raw, ccxt, coin, quote = normalize_symbol(symbol_input)

        if not raw:
            console.print("[red]無法識別的交易對格式[/]")
            Prompt.ask("按 Enter 繼續")
            return

        if raw in self.config.symbols:
            console.print(f"[yellow]{raw} 已存在[/]")
            Prompt.ask("按 Enter 繼續")
            return

        take_profit = FloatPrompt.ask("止盈間距 (%)", default=0.4) / 100
        grid_spacing = FloatPrompt.ask("補倉間距 (%)", default=0.6) / 100
        quantity = FloatPrompt.ask("每單數量", default=3.0)
        leverage = IntPrompt.ask("槓桿", default=20)

        # 動態倍數設定
        console.print(f"\n[dim]持倉控制 (基於每單數量 {quantity} 自動計算)[/]")
        limit_mult = FloatPrompt.ask("加倍倍數 (幾單後止盈加倍)", default=5.0)
        threshold_mult = FloatPrompt.ask("裝死倍數 (幾單後停止補倉)", default=20.0)
        console.print(f"[dim]→ 止盈加倍閾值: {quantity * limit_mult:.1f}, 裝死閾值: {quantity * threshold_mult:.1f}[/]")

        self.config.symbols[raw] = SymbolConfig(
            symbol=raw,
            ccxt_symbol=ccxt,
            enabled=True,
            take_profit_spacing=take_profit,
            grid_spacing=grid_spacing,
            initial_quantity=quantity,
            leverage=leverage,
            limit_multiplier=limit_mult,
            threshold_multiplier=threshold_mult
        )

        self.config.save()
        console.print(f"[green]已新增 {raw}[/]")
        Prompt.ask("按 Enter 繼續")

    def edit_symbol(self):
        if not self.config.symbols:
            console.print("[yellow]沒有可編輯的交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return

        symbols = list(self.config.symbols.keys())
        console.print("輸入序號編輯:")
        idx = IntPrompt.ask("序號", default=1) - 1

        if idx < 0 or idx >= len(symbols):
            console.print("[red]無效序號[/]")
            Prompt.ask("按 Enter 繼續")
            return

        key = symbols[idx]
        cfg = self.config.symbols[key]

        console.print(f"\n編輯 [cyan]{cfg.symbol}[/]")
        cfg.take_profit_spacing = FloatPrompt.ask(
            f"止盈間距 (%) [當前: {cfg.take_profit_spacing*100:.2f}]",
            default=cfg.take_profit_spacing * 100
        ) / 100
        cfg.grid_spacing = FloatPrompt.ask(
            f"補倉間距 (%) [當前: {cfg.grid_spacing*100:.2f}]",
            default=cfg.grid_spacing * 100
        ) / 100
        cfg.initial_quantity = FloatPrompt.ask(
            f"每單數量 [當前: {cfg.initial_quantity}]",
            default=cfg.initial_quantity
        )
        cfg.leverage = IntPrompt.ask(
            f"槓桿 [當前: {cfg.leverage}]",
            default=cfg.leverage
        )

        # 動態倍數設定
        console.print(f"\n[dim]持倉控制 (基於每單數量 {cfg.initial_quantity} 自動計算)[/]")
        cfg.limit_multiplier = FloatPrompt.ask(
            f"加倍倍數 (幾單後止盈加倍) [當前: {cfg.limit_multiplier}]",
            default=cfg.limit_multiplier
        )
        cfg.threshold_multiplier = FloatPrompt.ask(
            f"裝死倍數 (幾單後停止補倉) [當前: {cfg.threshold_multiplier}]",
            default=cfg.threshold_multiplier
        )
        console.print(f"[dim]→ 止盈加倍閾值: {cfg.position_limit:.1f}, 裝死閾值: {cfg.position_threshold:.1f}[/]")

        self.config.save()
        console.print("[green]已更新[/]")

        # 如果交易運行中，即時更新 bot 的配置
        if self._trading_active and self.bot:
            self.bot.config = self.config
            console.print("[cyan]✓ 配置已即時套用到運行中的交易[/]")

        Prompt.ask("按 Enter 繼續")

    def delete_symbol(self):
        if not self.config.symbols:
            console.print("[yellow]沒有可刪除的交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return

        symbols = list(self.config.symbols.keys())
        console.print("輸入序號刪除:")
        idx = IntPrompt.ask("序號", default=1) - 1

        if idx < 0 or idx >= len(symbols):
            console.print("[red]無效序號[/]")
            Prompt.ask("按 Enter 繼續")
            return

        key = symbols[idx]
        if Confirm.ask(f"確定刪除 {key}?"):
            del self.config.symbols[key]
            self.config.save()
            console.print("[green]已刪除[/]")

        Prompt.ask("按 Enter 繼續")

    def toggle_symbol(self):
        if not self.config.symbols:
            console.print("[yellow]沒有交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return

        symbols = list(self.config.symbols.keys())
        console.print("輸入序號切換:")
        idx = IntPrompt.ask("序號", default=1) - 1

        if idx < 0 or idx >= len(symbols):
            console.print("[red]無效序號[/]")
            Prompt.ask("按 Enter 繼續")
            return

        key = symbols[idx]
        cfg = self.config.symbols[key]
        cfg.enabled = not cfg.enabled
        self.config.save()

        status = "啟用" if cfg.enabled else "停用"
        console.print(f"[green]{key} 已{status}[/]")

        # 如果交易運行中，提示需要重啟才能生效
        if self._trading_active:
            console.print("[yellow]注意: 交易對啟用/停用需要重啟交易才能生效[/]")

        Prompt.ask("按 Enter 繼續")

    def start_trading(self):
        """啟動背景交易"""
        if not self.config.api_key:
            console.print("[red]請先設定 API[/]")
            Prompt.ask("按 Enter 繼續")
            return

        enabled = [s for s in self.config.symbols.values() if s.enabled]
        if not enabled:
            console.print("[red]沒有啟用的交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return

        if self._trading_active:
            console.print("[yellow]交易已在運行中[/]")
            Prompt.ask("按 Enter 繼續")
            return

        console.print("[bold]啟動 MAX 網格交易...[/]\n")

        # 創建 bot
        self.bot = MaxGridBot(self.config)

        def run_bot_thread():
            """在背景線程運行 bot"""
            self.bot_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.bot_loop)
            try:
                self.bot_loop.run_until_complete(self.bot.run())
            except Exception as e:
                logger.error(f"Bot 運行錯誤: {e}")
            finally:
                self.bot_loop.close()
                self._trading_active = False

        # 啟動背景線程
        self.bot_thread = threading.Thread(target=run_bot_thread, daemon=True)
        self.bot_thread.start()

        # 等待 bot 初始化 (API 連接需要時間)
        with console.status("[bold cyan]連接交易所...[/]"):
            for _ in range(100):  # 最多等待 10 秒
                if self.bot.state.running:
                    break
                time.sleep(0.1)

        if self.bot.state.running:
            self._trading_active = True
            console.print("[bold green]✓ 交易已在背景啟動！[/]\n")
            console.print("[dim]可以返回主選單管理設定，交易會持續運行[/]")
            console.print("[dim]選擇「1」查看交易面板，「s」停止交易[/]\n")
        else:
            # 檢查線程是否還活著 (可能只是初始化慢)
            if self.bot_thread and self.bot_thread.is_alive():
                console.print("[yellow]初始化較慢，請稍等...[/]")
                # 再等 10 秒
                for _ in range(100):
                    if self.bot.state.running:
                        break
                    time.sleep(0.1)

                if self.bot.state.running:
                    self._trading_active = True
                    console.print("[bold green]✓ 交易已在背景啟動！[/]\n")
                else:
                    console.print("[red]Bot 啟動超時，請檢查網絡連接[/]")
                    self.bot = None
            else:
                console.print("[red]Bot 啟動失敗，請檢查日誌[/]")
                self.bot = None

        Prompt.ask("按 Enter 繼續")

    def stop_trading(self):
        """停止背景交易"""
        if not self._trading_active or not self.bot:
            console.print("[yellow]沒有運行中的交易[/]")
            return

        console.print("[bold yellow]正在停止交易...[/]")

        # 停止 bot
        if self.bot_loop and self.bot_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.bot.stop(), self.bot_loop)

        # 等待線程結束
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=5)

        self._trading_active = False
        self.bot = None
        self.bot_thread = None
        self.bot_loop = None

        console.print("[green]✓ 交易已停止[/]")
        Prompt.ask("按 Enter 繼續")

    def view_trading_panel(self):
        """查看交易面板 (可按任意鍵返回主選單)"""
        if not self._trading_active or not self.bot:
            console.print("[yellow]沒有運行中的交易[/]")
            Prompt.ask("按 Enter 繼續")
            return

        ui = TerminalUI(self.config, self.bot.state, self.bot)

        console.print("[dim]按 Ctrl+C 返回主選單 (交易會繼續運行)[/]\n")

        try:
            with Live(ui.create_layout(), console=console, refresh_per_second=2) as live:
                while self._trading_active and self.bot.state.running:
                    live.update(ui.create_layout())
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        console.print("\n[dim]返回主選單...[/]")

    def reload_config(self):
        """重新載入配置並套用到運行中的 bot"""
        # 保存当前的 API 凭证（如果有的话）
        old_api_key = self.config.api_key if self.config else ""
        old_api_secret = self.config.api_secret if self.config else ""
        
        # 重新加载配置
        self.config = GlobalConfig.load()
        
        # 如果新配置中没有 API 凭证，但旧配置有，则保留旧的
        if not self.config.api_key and old_api_key:
            self.config.api_key = old_api_key
        if not self.config.api_secret and old_api_secret:
            self.config.api_secret = old_api_secret

        if self._trading_active and self.bot:
            # 更新 bot 的配置引用
            self.bot.config = self.config
            console.print("[green]✓ 配置已重新載入[/]")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              程式入口                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    menu = MainMenu()
    menu.main_menu()
