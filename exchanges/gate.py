# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Gate.io Adapter
===============
Gate.io 交易所適配器實作

WebSocket 訊息格式:
- Ticker: {"channel": "futures.tickers", "result": [...]}
- Order: {"channel": "futures.orders", "result": [...]}
- Position: {"channel": "futures.positions", "result": [...]}
"""

import json
import logging
import time
import hmac
import hashlib
from typing import Optional, Dict, List

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

# Gate.io Futures WebSocket URLs
GATE_WS_MAINNET = "wss://fx-ws.gateio.ws/v4/ws/usdt"
GATE_WS_TESTNET = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"

# Gate.io 心跳間隔 (15 秒)
GATE_PING_INTERVAL = 15


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Gate.io Adapter                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GateAdapter(ExchangeAdapter):
    """
    Gate.io 期貨交易所適配器

    Features:
        - CCXT gate 封裝
        - WebSocket 訂閱 (ticker, order, position)
        - 標準化 WebSocket 消息解析
    """

    def __init__(self):
        super().__init__()
        self._testnet = False
        self._api_key: str = ""
        self._api_secret: str = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # 基本資訊
    # ═══════════════════════════════════════════════════════════════════════════

    def get_exchange_name(self) -> str:
        return "gate"

    def get_display_name(self) -> str:
        return "Gate.io"

    # ═══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════════════

    def init_exchange(self, api_key: str, api_secret: str,
                      testnet: bool = False) -> None:
        """初始化 Gate.io CCXT 實例"""
        self._testnet = testnet
        self._api_key = api_key
        self._api_secret = api_secret

        options = {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "swap"}  # USDT 永續
        }

        if testnet:
            options["sandbox"] = True

        self.exchange = ccxt.gate(options)
        self.exchange.options["defaultType"] = "swap"

        logger.info(f"[Gate.io] 交易所初始化完成 (testnet={testnet})")

    def load_markets(self) -> None:
        """載入市場資訊"""
        if not self.exchange:
            raise RuntimeError("請先呼叫 init_exchange()")

        self.exchange.load_markets(reload=False)
        self._markets_loaded = True
        logger.info(f"[Gate.io] 已載入 {len(self.exchange.markets)} 個市場")

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
                min_notional=1.0,  # Gate.io 最小名義價值
                tick_size=precision.get("price", 4),
                step_size=precision.get("amount", 0),
            )
        except Exception as e:
            logger.error(f"[Gate.io] 獲取 {symbol} 精度失敗: {e}")
            return PrecisionInfo(
                price_precision=4,
                amount_precision=0,
                min_quantity=0,
                min_notional=1.0,
            )

    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 CCXT 格式

        Examples:
            BTC_USDT -> BTC/USDT:USDT
            BTCUSDT -> BTC/USDT:USDT
        """
        raw = raw_symbol.upper().replace("/", "").replace(":", "").replace("_", "")

        # 嘗試匹配報價幣種
        for quote in ["USDC", "USDT"]:
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                return f"{base}/{quote}:{quote}"

        logger.warning(f"[Gate.io] 無法轉換符號: {raw_symbol}")
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 WebSocket 訂閱格式

        Gate.io 合約使用下劃線格式
        Examples:
            XRP/USDT:USDT -> XRP_USDT
            XRPUSDT -> XRP_USDT
        """
        # 先移除分隔符
        raw = raw_symbol.upper().replace("/", "").replace(":", "")

        # 嘗試匹配報價幣種並加入下劃線
        for quote in ["USDC", "USDT"]:
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                return f"{base}_{quote}"

        return raw_symbol

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 帳戶
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        """獲取帳戶餘額"""
        result = {}

        try:
            balance = self.exchange.fetch_balance({"type": "swap", "settle": "usdt"})

            for currency in ["USDC", "USDT"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(
                        currency=currency,
                        wallet_balance=float(info.get("total", 0) or 0),
                        available_balance=float(info.get("free", 0) or 0),
                    )
        except Exception as e:
            logger.error(f"[Gate.io] 獲取餘額失敗: {e}")

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
            logger.error(f"[Gate.io] 獲取持倉失敗: {e}")

        return result

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """設定槓桿"""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"[Gate.io] {symbol} 槓桿設為 {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"[Gate.io] 設置 {symbol} 槓桿失敗: {e}")
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
        """
        創建限價單

        Gate.io 合約使用 size 正負號表示方向:
        - 正數 = 做多 (long)
        - 負數 = 做空 (short)

        對於雙向持倉模式:
        - 開多: side=buy, size=正數
        - 平多: side=sell, size=正數, close=True
        - 開空: side=sell, size=負數
        - 平空: side=buy, size=負數, close=True
        """
        params = {"settle": "usdt"}

        # Gate.io 使用 size 正負號處理方向
        # 當 position_side 為 SHORT 時，amount 需要為負數
        actual_amount = amount
        if position_side == "SHORT":
            actual_amount = -abs(amount)
        elif position_side == "LONG":
            actual_amount = abs(amount)

        if reduce_only:
            params["reduceOnly"] = True
            params["close"] = True

        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=actual_amount,
            price=price,
            params=params
        )

        logger.info(f"[Gate.io] 限價單: {symbol} {side} {amount}@{price} (pos={position_side})")
        return order

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """創建市價單 (處理 position_side)"""
        params = {"settle": "usdt"}

        # Gate.io 使用 size 正負號處理方向
        actual_amount = amount
        if position_side == "SHORT":
            actual_amount = -abs(amount)
        elif position_side == "LONG":
            actual_amount = abs(amount)

        if reduce_only:
            params["reduceOnly"] = True
            params["close"] = True

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=actual_amount,
            params=params
        )

        logger.info(f"[Gate.io] 市價單: {symbol} {side} {amount} (pos={position_side})")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消訂單"""
        try:
            self.exchange.cancel_order(order_id, symbol, {"settle": "usdt"})
            return True
        except Exception as e:
            logger.warning(f"[Gate.io] 取消訂單失敗: {e}")
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """獲取未成交訂單"""
        try:
            params = {"settle": "usdt"}
            return self.exchange.fetch_open_orders(symbol, params=params)
        except Exception as e:
            logger.error(f"[Gate.io] 獲取掛單失敗: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 其他
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_funding_rate(self, symbol: str) -> float:
        """獲取資金費率"""
        try:
            funding = self.exchange.fetch_funding_rate(symbol, {"settle": "usdt"})
            return float(funding.get("fundingRate", 0) or 0)
        except Exception as e:
            logger.error(f"[Gate.io] 獲取資金費率失敗: {e}")
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket
    # ═══════════════════════════════════════════════════════════════════════════

    def get_websocket_url(self) -> str:
        """獲取 WebSocket 基礎 URL"""
        if self._testnet:
            return GATE_WS_TESTNET
        return GATE_WS_MAINNET

    async def start_user_stream(self) -> Optional[str]:
        """
        Gate.io 使用 channel-based 認證
        返回認證參數 JSON 字串
        """
        if not self._api_key or not self._api_secret:
            logger.error("[Gate.io] 缺少 API 憑證")
            return None

        # 生成認證參數
        timestamp = int(time.time())
        # Gate.io 使用 channel + timestamp 認證

        logger.info("[Gate.io] 已準備 WebSocket 認證參數")
        return json.dumps({"ready": True, "timestamp": timestamp})

    def _generate_channel_signature(self, channel: str, timestamp: int) -> str:
        """生成頻道訂閱簽名"""
        message = f"channel={channel}&event=subscribe&time={timestamp}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha512
        ).hexdigest()
        return signature

    async def keepalive_user_stream(self) -> None:
        """Gate.io 需要定期發送 ping"""
        pass

    def get_keepalive_interval(self) -> int:
        """獲取 keepalive 間隔 (秒)"""
        return GATE_PING_INTERVAL

    def build_stream_url(
        self,
        symbols: List[str],
        user_stream_key: Optional[str] = None
    ) -> str:
        """建構完整的 WebSocket 訂閱 URL"""
        return self.get_websocket_url()

    def get_subscription_message(self, symbols: List[str]) -> List[str]:
        """
        生成訂閱消息列表

        Gate.io 需要分別訂閱不同頻道

        Args:
            symbols: 要訂閱的交易對列表

        Returns:
            JSON 訂閱消息列表
        """
        messages = []
        timestamp = int(time.time())

        # 訂閱訂單 (私有)
        order_sign = self._generate_channel_signature("futures.orders", timestamp)
        messages.append(json.dumps({
            "time": timestamp,
            "channel": "futures.orders",
            "event": "subscribe",
            "payload": ["!all"],
            "auth": {
                "method": "api_key",
                "KEY": self._api_key,
                "SIGN": order_sign
            }
        }))

        # 訂閱持倉 (私有)
        pos_sign = self._generate_channel_signature("futures.positions", timestamp)
        messages.append(json.dumps({
            "time": timestamp,
            "channel": "futures.positions",
            "event": "subscribe",
            "payload": ["!all"],
            "auth": {
                "method": "api_key",
                "KEY": self._api_key,
                "SIGN": pos_sign
            }
        }))

        # 訂閱餘額 (私有)
        balance_sign = self._generate_channel_signature("futures.balances", timestamp)
        messages.append(json.dumps({
            "time": timestamp,
            "channel": "futures.balances",
            "event": "subscribe",
            "payload": ["usdt"],
            "auth": {
                "method": "api_key",
                "KEY": self._api_key,
                "SIGN": balance_sign
            }
        }))

        # 訂閱 ticker (公開)
        ws_symbols = [self.convert_symbol_to_ws(s) for s in symbols]
        messages.append(json.dumps({
            "time": timestamp,
            "channel": "futures.tickers",
            "event": "subscribe",
            "payload": ws_symbols
        }))

        return messages

    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        解析 WebSocket 原始消息

        Gate.io 消息格式:
            {"channel": "futures.tickers", "event": "update", "result": [...]}
        """
        try:
            data = json.loads(raw_message)

            # 處理系統消息
            event = data.get("event")
            if event in ["subscribe", "unsubscribe"]:
                logger.debug(f"[Gate.io] 系統消息: {event}")
                return None

            channel = data.get("channel", "")
            result = data.get("result", [])

            if not result:
                return None

            # Ticker 更新
            if channel == "futures.tickers":
                ticker = self._parse_ticker(result[0])
                if ticker:
                    return WSMessage(
                        msg_type=WSMessageType.TICKER,
                        symbol=ticker.symbol,
                        data=ticker
                    )

            # 訂單更新
            elif channel == "futures.orders":
                order = self._parse_order_update(result[0])
                if order:
                    return WSMessage(
                        msg_type=WSMessageType.ORDER_UPDATE,
                        symbol=order.symbol,
                        data=order
                    )

            # 持倉更新
            elif channel == "futures.positions":
                account = self._parse_position_update(result)
                if account:
                    return WSMessage(
                        msg_type=WSMessageType.ACCOUNT_UPDATE,
                        data=account
                    )

            # 餘額更新
            elif channel == "futures.balances":
                account = self._parse_balance_update(result)
                if account:
                    return WSMessage(
                        msg_type=WSMessageType.ACCOUNT_UPDATE,
                        data=account
                    )

            return None

        except Exception as e:
            logger.error(f"[Gate.io] 解析 WS 消息失敗: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 內部解析方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _parse_ticker(self, data: dict) -> Optional[TickerUpdate]:
        """
        解析 Gate.io ticker 消息

        格式: {
            "contract": "BTC_USDT",
            "last": "50000.00",
            "highest_bid": "49999.00",
            "lowest_ask": "50001.00"
        }
        """
        try:
            return TickerUpdate(
                symbol=data.get("contract", ""),
                price=float(data.get("last", 0)),
                bid=float(data.get("highest_bid", 0) or data.get("bid1_price", 0)),
                ask=float(data.get("lowest_ask", 0) or data.get("ask1_price", 0)),
                timestamp=time.time(),
            )
        except Exception:
            return None

    def _parse_order_update(self, order_data: dict) -> Optional[OrderUpdate]:
        """
        解析 Gate.io 訂單更新

        格式: {
            "contract": "BTC_USDT",
            "id": "123456",
            "size": 1,
            "left": 0,
            "price": "50000",
            "fill_price": "50000",
            "is_reduce_only": false,
            "is_close": false,
            "status": "finished"
        }
        """
        try:
            # 根據 size 判斷方向
            size = float(order_data.get("size", 0))
            side = "BUY" if size > 0 else "SELL"

            # 轉換 status
            status_map = {
                "open": "NEW",
                "finished": "FILLED",
                "cancelled": "CANCELED",
            }
            raw_status = order_data.get("status", "").lower()
            status = status_map.get(raw_status, "UNKNOWN")

            # 計算 filled
            left = float(order_data.get("left", 0))
            filled = abs(size) - abs(left)

            return OrderUpdate(
                symbol=order_data.get("contract", ""),
                order_id=str(order_data.get("id", "")),
                side=side,
                position_side="BOTH",  # Gate.io 不分持倉方向
                status=status,
                order_type="LIMIT",  # 預設
                quantity=abs(size),
                filled_quantity=filled,
                price=float(order_data.get("price", 0)),
                avg_price=float(order_data.get("fill_price", 0) or 0),
                realized_pnl=float(order_data.get("pnl", 0) or 0),
                commission=float(order_data.get("fee", 0) or 0),
                is_reduce_only=order_data.get("is_reduce_only", False),
                timestamp=float(order_data.get("finish_time", 0) or order_data.get("create_time", 0)),
            )
        except Exception:
            return None

    def _parse_position_update(self, positions: list) -> Optional[AccountUpdate]:
        """
        解析 Gate.io 持倉更新

        格式: [{
            "contract": "BTC_USDT",
            "size": 1,
            "entry_price": "50000",
            "unrealised_pnl": "10",
            "leverage": 10
        }]
        """
        try:
            result = []
            for pos in positions:
                size = float(pos.get("size", 0))
                if size == 0:
                    continue

                # Gate.io 用正負數表示方向
                if size > 0:
                    position_side = "LONG"
                else:
                    position_side = "SHORT"

                result.append(PositionUpdate(
                    symbol=pos.get("contract", ""),
                    position_side=position_side,
                    quantity=abs(size),
                    entry_price=float(pos.get("entry_price", 0)),
                    unrealized_pnl=float(pos.get("unrealised_pnl", 0)),
                    leverage=int(pos.get("leverage", 1) or 1),
                ))

            return AccountUpdate(
                positions=result,
                balances=[],
                timestamp=time.time(),
            )
        except Exception:
            return None

    def _parse_balance_update(self, balance_data: list) -> Optional[AccountUpdate]:
        """
        解析 Gate.io 餘額更新

        格式: [{
            "currency": "USDT",
            "balance": "1000",
            "available": "900"
        }]
        """
        try:
            balances = []
            for bal in balance_data:
                currency = bal.get("currency", "").upper()
                if currency not in ["USDC", "USDT"]:
                    continue

                balances.append(BalanceUpdate(
                    currency=currency,
                    wallet_balance=float(bal.get("balance", 0)),
                    available_balance=float(bal.get("available", 0)),
                ))

            return AccountUpdate(
                positions=[],
                balances=balances,
                timestamp=time.time(),
            )
        except Exception:
            return None
