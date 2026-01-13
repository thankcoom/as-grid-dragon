# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Binance Adapter
===============
幣安交易所適配器實作

WebSocket 訊息格式:
- Ticker: {"c": price, "b": bid, "a": ask}
- Order: {"X": status, "S": side, "q": qty, "L": price, "rp": pnl}
- Account: {"P": positions, "B": balances}
"""

import json
import logging
import time
from typing import Optional, Dict, List, Any

import ccxt

from .base import (
    ExchangeAdapter,
    TickerUpdate,
    OrderUpdate,
    PositionUpdate,
    BalanceUpdate,
    AccountUpdate,
    PrecisionInfo,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger("as_grid_max")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              常量定義                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Binance Futures WebSocket URLs
BINANCE_WS_MAINNET = "wss://fstream.binance.com/ws"
BINANCE_WS_TESTNET = "wss://stream.binancefuture.com/ws"

# Listen Key 有效期 (60 分鐘)
LISTEN_KEY_KEEPALIVE_INTERVAL = 1800  # 30 分鐘 keepalive


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Binance Adapter                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BinanceAdapter(ExchangeAdapter):
    """
    幣安期貨交易所適配器

    Features:
        - CCXT binance 封裝
        - User Data Stream (Listen Key)
        - 標準化 WebSocket 消息解析
    """

    def __init__(self):
        super().__init__()
        self._testnet = False
        self._listen_key: Optional[str] = None
        self._listen_key_time: float = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 基本資訊
    # ═══════════════════════════════════════════════════════════════════════════

    def get_exchange_name(self) -> str:
        return "binance"

    def get_display_name(self) -> str:
        return "Binance"

    # ═══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════════════

    def init_exchange(self, api_key: str, api_secret: str,
                      testnet: bool = False) -> None:
        """初始化 Binance CCXT 實例"""
        self._testnet = testnet

        options = {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "future"}
        }

        if testnet:
            options["urls"] = {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                }
            }

        self.exchange = ccxt.binance(options)
        self.exchange.options["defaultType"] = "future"

        logger.info(f"[Binance] 交易所初始化完成 (testnet={testnet})")

    def load_markets(self) -> None:
        """載入市場資訊"""
        if not self.exchange:
            raise RuntimeError("請先呼叫 init_exchange()")

        self.exchange.load_markets(reload=False)
        self._markets_loaded = True
        logger.info(f"[Binance] 已載入 {len(self.exchange.markets)} 個市場")

    # ═══════════════════════════════════════════════════════════════════════════
    # 市場資訊
    # ═══════════════════════════════════════════════════════════════════════════

    def get_precision(self, symbol: str) -> PrecisionInfo:
        """獲取交易對精度資訊"""
        if not self._markets_loaded:
            raise RuntimeError("請先呼叫 load_markets()")

        try:
            market = self.exchange.market(symbol)
            precision = market.get("precision", {})
            limits = market.get("limits", {})

            return PrecisionInfo(
                price_precision=precision.get("price", 4),
                amount_precision=precision.get("amount", 0),
                min_quantity=limits.get("amount", {}).get("min", 0),
                min_notional=5.0,  # Binance 最小名義價值
                tick_size=precision.get("price", 4),
                step_size=precision.get("amount", 0),
            )
        except Exception as e:
            logger.error(f"[Binance] 獲取 {symbol} 精度失敗: {e}")
            return PrecisionInfo(
                price_precision=4,
                amount_precision=0,
                min_quantity=0,
                min_notional=5.0,
            )

    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 CCXT 格式

        Examples:
            XRPUSDC -> XRP/USDC:USDC
            BTCUSDT -> BTC/USDT:USDT
        """
        raw = raw_symbol.upper().replace("/", "").replace(":", "")

        # 嘗試匹配報價幣種
        for quote in ["USDC", "USDT", "BUSD"]:
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                return f"{base}/{quote}:{quote}"

        # 無法識別，返回原始
        logger.warning(f"[Binance] 無法轉換符號: {raw_symbol}")
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 WebSocket 訂閱格式

        Examples:
            XRP/USDC:USDC -> xrpusdc
            XRPUSDC -> xrpusdc
        """
        # 移除所有分隔符並轉小寫
        ws_sym = raw_symbol.replace("/", "").replace(":", "")
        return ws_sym.lower()

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 帳戶
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        """獲取帳戶餘額"""
        result = {}

        try:
            balance = self.exchange.fetch_balance()

            for currency in ["USDC", "USDT", "BNB"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(
                        currency=currency,
                        wallet_balance=float(info.get("total", 0) or 0),
                        available_balance=float(info.get("free", 0) or 0),
                    )
        except Exception as e:
            logger.error(f"[Binance] 獲取餘額失敗: {e}")

        return result

    def fetch_positions(self) -> List[PositionUpdate]:
        """獲取所有持倉"""
        result = []

        try:
            positions = self.exchange.fetch_positions()

            for pos in positions:
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts == 0:
                    continue

                side = pos.get("side", "").upper()
                if side not in ["LONG", "SHORT"]:
                    continue

                result.append(PositionUpdate(
                    symbol=pos.get("symbol", ""),
                    position_side=side,
                    quantity=abs(contracts),
                    entry_price=float(pos.get("entryPrice", 0) or 0),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                    leverage=int(pos.get("leverage", 1) or 1),
                ))
        except Exception as e:
            logger.error(f"[Binance] 獲取持倉失敗: {e}")

        return result

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """設定槓桿"""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"[Binance] {symbol} 槓桿設為 {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"[Binance] 設置 {symbol} 槓桿失敗: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 訂單
    # ═══════════════════════════════════════════════════════════════════════════

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """創建限價單"""
        params = {}

        if reduce_only:
            params["reduceOnly"] = True

        if position_side != "BOTH":
            params["positionSide"] = position_side

        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=amount,
            price=price,
            params=params
        )

        logger.info(f"[Binance] 限價單: {symbol} {side} {amount}@{price}")
        return order

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """創建市價單"""
        params = {}

        if reduce_only:
            params["reduceOnly"] = True

        if position_side != "BOTH":
            params["positionSide"] = position_side

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=amount,
            params=params
        )

        logger.info(f"[Binance] 市價單: {symbol} {side} {amount}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消訂單"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"[Binance] 取消訂單失敗: {e}")
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """獲取未成交訂單"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"[Binance] 獲取掛單失敗: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 其他
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_funding_rate(self, symbol: str) -> float:
        """獲取資金費率"""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0) or 0)
        except Exception as e:
            logger.error(f"[Binance] 獲取資金費率失敗: {e}")
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket
    # ═══════════════════════════════════════════════════════════════════════════

    def get_websocket_url(self) -> str:
        """獲取 WebSocket 基礎 URL"""
        if self._testnet:
            return BINANCE_WS_TESTNET
        return BINANCE_WS_MAINNET

    async def start_user_stream(self) -> Optional[str]:
        """
        啟動用戶數據流，獲取 Listen Key

        Binance 需要 Listen Key 來訂閱用戶數據 (訂單、帳戶更新)
        """
        try:
            response = self.exchange.fapiPrivatePostListenKey()
            self._listen_key = response.get("listenKey")
            self._listen_key_time = time.time()
            logger.info("[Binance] 獲取 Listen Key 成功")
            return self._listen_key
        except Exception as e:
            logger.error(f"[Binance] 獲取 Listen Key 失敗: {e}")
            return None

    async def keepalive_user_stream(self) -> None:
        """延長 Listen Key 有效期"""
        try:
            self.exchange.fapiPrivatePutListenKey()
            self._listen_key_time = time.time()
            logger.info("[Binance] Listen Key 已延長")
        except Exception as e:
            logger.error(f"[Binance] 延長 Listen Key 失敗: {e}")
            # 嘗試重新獲取
            await self.start_user_stream()

    def get_keepalive_interval(self) -> int:
        """獲取 keepalive 間隔 (秒)"""
        return LISTEN_KEY_KEEPALIVE_INTERVAL

    def build_stream_url(
        self,
        symbols: List[str],
        user_stream_key: Optional[str] = None
    ) -> str:
        """
        建構完整的 WebSocket 訂閱 URL

        Args:
            symbols: 要訂閱的交易對列表 (原始格式, e.g., ["XRPUSDC"])
            user_stream_key: Listen Key

        Returns:
            完整 URL, e.g.,
            wss://fstream.binance.com/ws/stream?streams=xrpusdc@ticker/listenKey
        """
        streams = []

        # 添加 ticker 訂閱
        for symbol in symbols:
            ws_sym = self.convert_symbol_to_ws(symbol)
            streams.append(f"{ws_sym}@ticker")

        # 添加用戶數據流
        if user_stream_key:
            streams.append(user_stream_key)

        base_url = self.get_websocket_url()
        return f"{base_url}/stream?streams=" + "/".join(streams)

    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        解析 WebSocket 原始消息

        Binance 消息格式:
            Combined Stream: {"stream": "xrpusdc@ticker", "data": {...}}
            User Data: {"e": "ORDER_TRADE_UPDATE", ...}
        """
        try:
            data = json.loads(raw_message)

            # Combined Stream 格式
            if "stream" in data and "data" in data:
                stream = data["stream"]
                payload = data["data"]

                # Ticker 更新
                if "@ticker" in stream:
                    ticker = self._parse_ticker(payload)
                    if ticker:
                        return WSMessage(
                            msg_type=WSMessageType.TICKER,
                            symbol=ticker.symbol,
                            data=ticker
                        )

                # User Data Stream
                elif stream == self._listen_key:
                    return self._parse_user_data(payload)

            # 直接的用戶數據事件
            elif "e" in data:
                return self._parse_user_data(data)

            return None

        except Exception as e:
            logger.error(f"[Binance] 解析 WS 消息失敗: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 內部解析方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _parse_ticker(self, data: dict) -> Optional[TickerUpdate]:
        """
        解析 Binance ticker 消息

        格式: {
            "s": "XRPUSDC",  // Symbol
            "c": "0.5432",   // Close price
            "b": "0.5431",   // Best bid
            "a": "0.5433",   // Best ask
            "E": 1234567890  // Event time
        }
        """
        try:
            return TickerUpdate(
                symbol=data.get("s", ""),
                price=float(data.get("c", 0)),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                timestamp=float(data.get("E", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_user_data(self, data: dict) -> Optional[WSMessage]:
        """解析用戶數據事件"""
        event_type = data.get("e")

        if event_type == "ORDER_TRADE_UPDATE":
            order = self._parse_order_update(data.get("o", {}))
            if order:
                return WSMessage(
                    msg_type=WSMessageType.ORDER_UPDATE,
                    symbol=order.symbol,
                    data=order
                )

        elif event_type == "ACCOUNT_UPDATE":
            account = self._parse_account_update(data.get("a", {}))
            if account:
                return WSMessage(
                    msg_type=WSMessageType.ACCOUNT_UPDATE,
                    data=account
                )

        return None

    def _parse_order_update(self, order_data: dict) -> Optional[OrderUpdate]:
        """
        解析 Binance 訂單更新

        格式: {
            "s": "XRPUSDC",   // Symbol
            "S": "BUY",       // Side
            "X": "FILLED",    // Status
            "o": "LIMIT",     // Order type
            "q": "10",        // Quantity
            "z": "10",        // Filled qty
            "p": "0.54",      // Price
            "L": "0.5432",    // Last filled price
            "rp": "0.05",     // Realized PnL
            "n": "0.001",     // Commission
            "R": false,       // Reduce only
            "ps": "LONG",     // Position side
        }
        """
        try:
            return OrderUpdate(
                symbol=order_data.get("s", ""),
                order_id=str(order_data.get("i", "")),
                side=order_data.get("S", ""),
                position_side=order_data.get("ps", "BOTH"),
                status=order_data.get("X", ""),
                order_type=order_data.get("o", ""),
                quantity=float(order_data.get("q", 0)),
                filled_quantity=float(order_data.get("z", 0)),
                price=float(order_data.get("p", 0)),
                avg_price=float(order_data.get("L", 0)),
                realized_pnl=float(order_data.get("rp", 0)),
                commission=float(order_data.get("n", 0)),
                is_reduce_only=order_data.get("R", False),
                timestamp=float(order_data.get("T", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_account_update(self, account_data: dict) -> Optional[AccountUpdate]:
        """
        解析 Binance 帳戶更新

        格式: {
            "P": [  // Positions
                {"s": "XRPUSDC", "ps": "LONG", "pa": "10", "up": "0.05"}
            ],
            "B": [  // Balances
                {"a": "USDC", "wb": "100", "cw": "90"}
            ]
        }
        """
        try:
            positions = []
            for pos in account_data.get("P", []):
                amount = float(pos.get("pa", 0))
                if amount == 0:
                    continue

                positions.append(PositionUpdate(
                    symbol=pos.get("s", ""),
                    position_side=pos.get("ps", "").upper(),
                    quantity=abs(amount),
                    entry_price=float(pos.get("ep", 0)),
                    unrealized_pnl=float(pos.get("up", 0)),
                ))

            balances = []
            for bal in account_data.get("B", []):
                asset = bal.get("a", "")
                if asset not in ["USDC", "USDT", "BNB"]:
                    continue

                balances.append(BalanceUpdate(
                    currency=asset,
                    wallet_balance=float(bal.get("wb", 0)),
                    available_balance=float(bal.get("cw", 0)),
                ))

            return AccountUpdate(
                positions=positions,
                balances=balances,
                timestamp=time.time(),
            )
        except Exception:
            return None
