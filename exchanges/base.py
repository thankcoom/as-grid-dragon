# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Exchange Adapter Base
=====================
交易所抽象介面定義

所有交易所適配器都必須實作此介面，
確保 GridBot 可以無縫切換不同交易所。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              標準化數據結構                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class TickerUpdate:
    """
    標準化的 Ticker 更新

    所有交易所的 ticker 數據都會轉換為此格式
    """
    symbol: str           # 原始交易對符號 (e.g., "XRPUSDC")
    price: float          # 最新價格
    bid: float            # 買一價
    ask: float            # 賣一價
    timestamp: float = 0  # Unix 時間戳


@dataclass
class OrderUpdate:
    """
    標準化的訂單更新

    所有交易所的訂單事件都會轉換為此格式
    """
    symbol: str           # 交易對
    order_id: str         # 訂單 ID
    side: str             # BUY / SELL
    position_side: str    # LONG / SHORT / BOTH
    status: str           # FILLED / PARTIALLY_FILLED / CANCELED / NEW / EXPIRED
    order_type: str       # LIMIT / MARKET
    quantity: float       # 訂單數量
    filled_quantity: float  # 已成交數量
    price: float          # 訂單價格
    avg_price: float      # 平均成交價
    realized_pnl: float   # 已實現盈虧
    commission: float     # 手續費
    is_reduce_only: bool  # 是否為減倉單
    timestamp: float = 0  # Unix 時間戳


@dataclass
class PositionUpdate:
    """
    標準化的持倉更新
    """
    symbol: str           # 交易對
    position_side: str    # LONG / SHORT
    quantity: float       # 持倉數量 (正數)
    entry_price: float    # 入場均價
    unrealized_pnl: float # 未實現盈虧
    leverage: int = 1     # 槓桿


@dataclass
class BalanceUpdate:
    """
    標準化的餘額更新
    """
    currency: str         # 幣種 (USDC, USDT, etc.)
    wallet_balance: float # 錢包餘額
    available_balance: float  # 可用餘額
    unrealized_pnl: float = 0  # 未實現盈虧


@dataclass
class AccountUpdate:
    """
    標準化的帳戶更新 (包含持倉和餘額)
    """
    positions: List[PositionUpdate] = field(default_factory=list)
    balances: List[BalanceUpdate] = field(default_factory=list)
    timestamp: float = 0


@dataclass
class PrecisionInfo:
    """
    交易對精度資訊
    """
    price_precision: int      # 價格小數位數
    amount_precision: int     # 數量小數位數
    min_quantity: float       # 最小下單數量
    min_notional: float       # 最小名義價值
    tick_size: float = 0      # 價格最小變動單位
    step_size: float = 0      # 數量最小變動單位


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              WebSocket 消息類型                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class WSMessageType:
    """WebSocket 消息類型常量"""
    TICKER = "ticker"
    ORDER_UPDATE = "order_update"
    ACCOUNT_UPDATE = "account_update"
    UNKNOWN = "unknown"


@dataclass
class WSMessage:
    """
    標準化的 WebSocket 消息
    """
    msg_type: str         # WSMessageType
    symbol: Optional[str] = None
    data: Any = None      # TickerUpdate / OrderUpdate / AccountUpdate


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Exchange Adapter 抽象類                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ExchangeAdapter(ABC):
    """
    交易所適配器抽象基類

    所有交易所的實作都必須繼承此類並實作所有抽象方法。
    這確保了 GridBot 可以透過統一介面與任何交易所互動。

    使用方式:
        adapter = BinanceAdapter()
        adapter.init_exchange(api_key, api_secret)
        adapter.load_markets()

        # REST API
        positions = adapter.fetch_positions()
        order = adapter.create_limit_order(...)

        # WebSocket
        url = adapter.build_stream_url(symbols, user_key)
        msg = adapter.parse_ws_message(raw_message)
    """

    def __init__(self):
        self.exchange = None  # CCXT exchange instance
        self._markets_loaded = False

    # ═══════════════════════════════════════════════════════════════════════════
    # 基本資訊
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def get_exchange_name(self) -> str:
        """
        返回交易所名稱

        Returns:
            str: 交易所識別名稱 (e.g., "binance", "bybit", "okx")
        """
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """
        返回交易所顯示名稱

        Returns:
            str: 用於 UI 顯示的名稱 (e.g., "Binance", "Bybit", "OKX")
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def init_exchange(self, api_key: str, api_secret: str,
                      testnet: bool = False) -> None:
        """
        初始化 CCXT 交易所實例

        Args:
            api_key: API Key
            api_secret: API Secret
            testnet: 是否使用測試網
        """
        pass

    @abstractmethod
    def load_markets(self) -> None:
        """
        載入市場資訊

        必須在其他 API 調用之前執行
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 市場資訊
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def get_precision(self, symbol: str) -> PrecisionInfo:
        """
        獲取交易對精度資訊

        Args:
            symbol: CCXT 格式交易對 (e.g., "XRP/USDC:USDC")

        Returns:
            PrecisionInfo: 精度資訊
        """
        pass

    @abstractmethod
    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 CCXT 格式

        Args:
            raw_symbol: 原始符號 (e.g., "XRPUSDC")

        Returns:
            str: CCXT 格式 (e.g., "XRP/USDC:USDC")
        """
        pass

    @abstractmethod
    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 WebSocket 訂閱格式

        Args:
            raw_symbol: 原始符號 (e.g., "XRPUSDC")

        Returns:
            str: WS 格式 (取決於交易所)
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 帳戶
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        """
        獲取帳戶餘額

        Returns:
            Dict[str, BalanceUpdate]: 幣種 -> 餘額資訊
        """
        pass

    @abstractmethod
    def fetch_positions(self) -> List[PositionUpdate]:
        """
        獲取所有持倉

        Returns:
            List[PositionUpdate]: 持倉列表
        """
        pass

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        設定槓桿

        Args:
            symbol: CCXT 格式交易對
            leverage: 槓桿倍數

        Returns:
            bool: 是否成功
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 訂單
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """
        創建限價單

        Args:
            symbol: CCXT 格式交易對
            side: "buy" 或 "sell"
            amount: 數量
            price: 價格
            position_side: "LONG", "SHORT", 或 "BOTH"
            reduce_only: 是否為減倉單

        Returns:
            Dict: 訂單資訊
        """
        pass

    @abstractmethod
    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """
        創建市價單

        Args:
            symbol: CCXT 格式交易對
            side: "buy" 或 "sell"
            amount: 數量
            position_side: "LONG", "SHORT", 或 "BOTH"
            reduce_only: 是否為減倉單

        Returns:
            Dict: 訂單資訊
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        取消訂單

        Args:
            order_id: 訂單 ID
            symbol: CCXT 格式交易對

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        獲取未成交訂單

        Args:
            symbol: CCXT 格式交易對 (None = 所有)

        Returns:
            List[Dict]: 訂單列表
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 其他
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        獲取資金費率

        Args:
            symbol: CCXT 格式交易對

        Returns:
            float: 資金費率
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def get_websocket_url(self) -> str:
        """
        獲取 WebSocket 基礎 URL

        Returns:
            str: WebSocket URL
        """
        pass

    @abstractmethod
    async def start_user_stream(self) -> Optional[str]:
        """
        啟動用戶數據流

        某些交易所 (如 Binance) 需要先獲取 listen key

        Returns:
            Optional[str]: stream key 或 None
        """
        pass

    @abstractmethod
    async def keepalive_user_stream(self) -> None:
        """
        保持用戶數據流連線

        某些交易所需要定期發送 keepalive 請求
        """
        pass

    @abstractmethod
    def get_keepalive_interval(self) -> int:
        """
        獲取 keepalive 間隔 (秒)

        Returns:
            int: 間隔秒數 (0 = 不需要 keepalive)
        """
        pass

    @abstractmethod
    def build_stream_url(
        self,
        symbols: List[str],
        user_stream_key: Optional[str] = None
    ) -> str:
        """
        建構完整的 WebSocket 訂閱 URL

        Args:
            symbols: 要訂閱的交易對列表 (原始格式)
            user_stream_key: 用戶數據流 key (如果有)

        Returns:
            str: 完整的 WebSocket URL
        """
        pass

    @abstractmethod
    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        解析 WebSocket 原始消息

        Args:
            raw_message: JSON 字串

        Returns:
            Optional[WSMessage]: 標準化消息，無法解析時返回 None
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 輔助方法 (可選覆寫)
    # ═══════════════════════════════════════════════════════════════════════════

    def round_price(self, symbol: str, price: float) -> float:
        """
        根據精度四捨五入價格
        """
        precision = self.get_precision(symbol)
        return round(price, precision.price_precision)

    def round_amount(self, symbol: str, amount: float) -> float:
        """
        根據精度四捨五入數量
        """
        precision = self.get_precision(symbol)
        return round(amount, precision.amount_precision)

    def is_initialized(self) -> bool:
        """
        檢查是否已初始化
        """
        return self.exchange is not None and self._markets_loaded
