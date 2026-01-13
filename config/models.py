# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
配置模型
========
所有 dataclass 配置類定義
"""

import json
from datetime import datetime
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Dict
from collections import deque

from constants import CONFIG_FILE


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              序列化 Mixin                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SerializableMixin:
    """
    Dataclass 序列化 Mixin

    提供標準化的 to_dict() 和 from_dict() 方法，
    減少每個 dataclass 中重複的序列化代碼。

    使用方式:
        @dataclass
        class MyConfig(SerializableMixin):
            field1: str = ""
            field2: int = 0
    """

    def to_dict(self) -> dict:
        """將 dataclass 轉換為字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """從字典建立 dataclass 實例，自動過濾無效欄位"""
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAX 增強配置                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MaxEnhancement(SerializableMixin):
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

    def is_feature_enabled(self, feature: str) -> bool:
        """檢查功能是否啟用 (考慮總開關)"""
        if not self.all_enhancements_enabled:
            return False
        return getattr(self, f"{feature}_enabled", False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Bandit 配置                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class BanditConfig(SerializableMixin):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              DGT 配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class DGTConfig(SerializableMixin):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              領先指標配置                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class LeadingIndicatorConfig(SerializableMixin):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              交易對配置                                         ║
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

    # 交易模式 - 用於優化器參數範圍選擇
    # "high_freq": 🚀 次高頻 (2-7天)
    # "swing": 📊 波動 (1週-1月)
    # "long_cycle": 🌊 大週期 (1月以上)
    trading_mode: str = "swing"

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
            "trading_mode": self.trading_mode,
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              風控配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class RiskConfig(SerializableMixin):
    """風控配置"""
    enabled: bool = True
    margin_threshold: float = 0.5
    trailing_start_profit: float = 5.0
    trailing_drawdown_pct: float = 0.10
    trailing_min_drawdown: float = 2.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              全局配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class GlobalConfig:
    """
    全局配置

    Attributes:
        exchange_type: 交易所類型 (binance, bybit, okx, etc.)
                       使用 exchanges.get_adapter() 支援的任何交易所
        testnet: 是否使用測試網
    """
    # === 交易所設定 ===
    exchange_type: str = "binance"    # 支援: binance, bybit, okx (需實作 adapter)
    testnet: bool = False             # 是否使用測試網

    # === API 憑證 ===
    api_key: str = ""
    api_secret: str = ""
    api_password: str = ""  # Bitget 專用 Passphrase

    # === 系統設定 ===
    sync_interval: float = 30.0

    # === 子配置 ===
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    max_enhancement: MaxEnhancement = field(default_factory=MaxEnhancement)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    dgt: DGTConfig = field(default_factory=DGTConfig)
    leading_indicator: LeadingIndicatorConfig = field(default_factory=LeadingIndicatorConfig)

    # === 內部狀態 ===
    # Story 1.4: 偵測舊版配置是否包含明文 API
    legacy_api_detected: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        """序列化為字典 (本地終端應用：儲存 API 金鑰到配置檔)"""
        return {
            "exchange_type": self.exchange_type,
            "testnet": self.testnet,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "api_password": self.api_password,
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
        """從字典建立 (向後兼容舊版配置)"""
        # 向後兼容: 如果有 websocket_url 但沒有 exchange_type，推斷交易所
        exchange_type = data.get("exchange_type", "binance")
        if "websocket_url" in data and "exchange_type" not in data:
            ws_url = data.get("websocket_url", "")
            if "binance" in ws_url:
                exchange_type = "binance"
            elif "bybit" in ws_url:
                exchange_type = "bybit"
            elif "okx" in ws_url:
                exchange_type = "okx"

        config = cls(
            exchange_type=exchange_type,
            testnet=data.get("testnet", False),
            api_key=data.get("api_key", ""),
            api_secret=data.get("api_secret", ""),
            api_password=data.get("api_password", ""),
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
        from rich.console import Console
        console = Console()
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
