# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Bybit Adapter
=============
Bybit 交易所適配器實作 (V5 Unified API)

WebSocket 訊息格式:
- Ticker: {"topic": "tickers.BTCUSDT", "data": {"lastPrice": "..."}}
- Order: {"topic": "order", "data": [...]}
- Position: {"topic": "position", "data": [...]}
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

# Bybit V5 WebSocket URLs
BYBIT_WS_MAINNET = "wss://stream.bybit.com/v5/private"
BYBIT_WS_PUBLIC_MAINNET = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/private"
BYBIT_WS_PUBLIC_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"

# Bybit WebSocket ping 間隔 (20 秒)
BYBIT_PING_INTERVAL = 20


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Bybit Adapter                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BybitAdapter(ExchangeAdapter):
    """
    Bybit 期貨交易所適配器 (V5 Unified API)

    Features:
        - CCXT bybit 封裝
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
        return "bybit"

    def get_display_name(self) -> str:
        return "Bybit"

    # ═══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════════════

    def init_exchange(self, api_key: str, api_secret: str,
                      testnet: bool = False, password: str = "") -> None:
        """初始化 Bybit CCXT 實例 (password 參數為介面統一，Bybit 不需要)"""
        self._testnet = testnet
        self._api_key = api_key
        self._api_secret = api_secret

        options = {
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "linear"}  # USDT 永續
        }

        if testnet:
            options["sandbox"] = True

        self.exchange = ccxt.bybit(options)
        self.exchange.options["defaultType"] = "linear"

        logger.info(f"[Bybit] 交易所初始化完成 (testnet={testnet})")

    def load_markets(self) -> None:
        """載入市場資訊"""
        if not self.exchange:
            raise RuntimeError("請先呼叫 init_exchange()")

        self.exchange.load_markets(reload=False)
        self._markets_loaded = True
        logger.info(f"[Bybit] 已載入 {len(self.exchange.markets)} 個市場")

    # ═══════════════════════════════════════════════════════════════════════════
    # 市場資訊
    # ═══════════════════════════════════════════════════════════════════════════

    def get_precision(self, symbol: str) -> PrecisionInfo:
        """
        獲取交易對精度資訊
        
        注意: CCXT 可能返回浮點精度 (如 0.0001)，需轉換為小數位數 (4)
        """
        import math
        
        if not self._markets_loaded:
            raise RuntimeError("請先呼叫 load_markets()")

        def _to_decimal_places(value):
            """將浮點精度轉換為小數位數 (如 0.0001 -> 4)"""
            if isinstance(value, float) and value > 0 and value < 1:
                return int(abs(math.log10(value)))
            return int(value) if value else 0

        try:
            market = self.exchange.market(symbol)
            precision = market.get("precision", {})
            limits = market.get("limits", {})

            price_prec = _to_decimal_places(precision.get("price", 4))
            amount_prec = _to_decimal_places(precision.get("amount", 0))
            min_qty = float(limits.get("amount", {}).get("min", 0) or 0)

            return PrecisionInfo(
                price_precision=price_prec,
                amount_precision=amount_prec,
                min_quantity=min_qty,
                min_notional=1.0,  # Bybit 最小名義價值
                tick_size=price_prec,
                step_size=amount_prec,
            )
        except Exception as e:
            logger.error(f"[Bybit] 獲取 {symbol} 精度失敗: {e}")
            return PrecisionInfo(
                price_precision=4,
                amount_precision=0,
                min_quantity=1,
                min_notional=1.0,
            )

    def convert_symbol_to_ccxt(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 CCXT 格式

        Examples:
            XRPUSDT -> XRP/USDT:USDT
            BTCUSDT -> BTC/USDT:USDT
        """
        raw = raw_symbol.upper().replace("/", "").replace(":", "")

        # 嘗試匹配報價幣種
        for quote in ["USDC", "USDT"]:
            if raw.endswith(quote):
                base = raw[:-len(quote)]
                return f"{base}/{quote}:{quote}"

        # 無法識別，返回原始
        logger.warning(f"[Bybit] 無法轉換符號: {raw_symbol}")
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 WebSocket 訂閱格式

        Bybit 使用大寫符號
        Examples:
            XRP/USDT:USDT -> XRPUSDT
            XRPUSDT -> XRPUSDT
        """
        # 處理 CCXT 格式 (移除 :USDT 後綴)
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":")[0]

        # 移除所有分隔符並轉大寫
        ws_sym = raw_symbol.replace("/", "").replace(":", "")
        return ws_sym.upper()

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 帳戶
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        """獲取帳戶餘額"""
        result = {}

        try:
            balance = self.exchange.fetch_balance({"type": "linear"})

            for currency in ["USDC", "USDT"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(
                        currency=currency,
                        wallet_balance=float(info.get("total", 0) or 0),
                        available_balance=float(info.get("free", 0) or 0),
                    )
        except Exception as e:
            logger.error(f"[Bybit] 獲取餘額失敗: {e}")

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
            logger.error(f"[Bybit] 獲取持倉失敗: {e}")

        return result

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """設定槓桿"""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"[Bybit] {symbol} 槓桿設為 {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"[Bybit] 設置 {symbol} 槓桿失敗: {e}")
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
        """創建限價單 (與終端版一致)"""
        params = {'reduce_only': reduce_only}

        # Bybit 使用 positionIdx 而非 positionSide
        # 0 = one-way mode, 1 = hedge-long, 2 = hedge-short
        if position_side == "LONG":
            params["positionIdx"] = 1
        elif position_side == "SHORT":
            params["positionIdx"] = 2
        else:
            params["positionIdx"] = 0

        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=amount,
            price=price,
            params=params
        )

        logger.info(f"[Bybit] 限價單: {symbol} {side} {amount}@{price}")
        return order

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str = "BOTH",
        reduce_only: bool = False
    ) -> Dict:
        """創建市價單 (與終端版一致)"""
        params = {'reduce_only': reduce_only}

        if position_side == "LONG":
            params["positionIdx"] = 1
        elif position_side == "SHORT":
            params["positionIdx"] = 2
        else:
            params["positionIdx"] = 0

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=amount,
            params=params
        )

        logger.info(f"[Bybit] 市價單: {symbol} {side} {amount}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消訂單"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"[Bybit] 取消訂單失敗: {e}")
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """獲取未成交訂單"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"[Bybit] 獲取掛單失敗: {e}")
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
            logger.error(f"[Bybit] 獲取資金費率失敗: {e}")
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket
    # ═══════════════════════════════════════════════════════════════════════════

    def get_websocket_url(self) -> str:
        """獲取 WebSocket 基礎 URL (私有頻道)"""
        if self._testnet:
            return BYBIT_WS_TESTNET
        return BYBIT_WS_MAINNET

    def get_public_websocket_url(self) -> str:
        """獲取公共 WebSocket URL"""
        if self._testnet:
            return BYBIT_WS_PUBLIC_TESTNET
        return BYBIT_WS_PUBLIC_MAINNET

    async def start_user_stream(self) -> Optional[str]:
        """
        Bybit V5 使用 API 簽名認證，不需要預先獲取 token
        返回認證參數 JSON 字串
        """
        if not self._api_key or not self._api_secret:
            logger.error("[Bybit] 缺少 API 憑證")
            return None

        # 生成認證參數
        expires = int((time.time() + 10000) * 1000)
        signature = self._generate_signature(expires)

        auth_params = {
            "op": "auth",
            "args": [self._api_key, expires, signature]
        }

        logger.info("[Bybit] 已準備 WebSocket 認證參數")
        return json.dumps(auth_params)

    def _generate_signature(self, expires: int) -> str:
        """生成 WebSocket 認證簽名"""
        param_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def keepalive_user_stream(self) -> None:
        """Bybit 使用心跳機制，需要定期發送 ping"""
        # Bybit WebSocket 自動處理 ping/pong
        pass

    def get_keepalive_interval(self) -> int:
        """獲取 keepalive 間隔 (秒)"""
        return BYBIT_PING_INTERVAL

    def build_stream_url(
        self,
        symbols: List[str],
        user_stream_key: Optional[str] = None
    ) -> str:
        """
        建構完整的 WebSocket 訂閱 URL

        Bybit V5 使用單獨連線，需要在連接後發送訂閱消息
        """
        # Bybit 使用單一 URL，訂閱在連接後進行
        return self.get_websocket_url()

    def get_subscription_message(self, symbols: List[str]) -> str:
        """
        生成訂閱消息

        Args:
            symbols: 要訂閱的交易對列表

        Returns:
            JSON 訂閱消息
        """
        # 訂閱 ticker, order, position, wallet
        topics = ["order", "position", "wallet"]

        # 添加 ticker 訂閱 (需要連接到公共頻道)
        for symbol in symbols:
            ws_sym = self.convert_symbol_to_ws(symbol)
            topics.append(f"tickers.{ws_sym}")

        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }

        return json.dumps(subscribe_msg)

    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        解析 WebSocket 原始消息

        Bybit V5 消息格式:
            {"topic": "tickers.BTCUSDT", "data": {...}}
            {"topic": "order", "data": [...]}
            {"topic": "position", "data": [...]}
        """
        try:
            data = json.loads(raw_message)

            # 處理系統消息
            if "op" in data:
                op = data.get("op")
                if op in ["subscribe", "auth", "pong"]:
                    logger.debug(f"[Bybit] 系統消息: {op}")
                    return None

            topic = data.get("topic", "")

            # Ticker 更新
            if topic.startswith("tickers."):
                ticker_data = data.get("data", {})
                ticker = self._parse_ticker(ticker_data)
                if ticker:
                    return WSMessage(
                        msg_type=WSMessageType.TICKER,
                        symbol=ticker.symbol,
                        data=ticker
                    )

            # 訂單更新
            elif topic == "order":
                orders = data.get("data", [])
                if orders:
                    order = self._parse_order_update(orders[0])
                    if order:
                        return WSMessage(
                            msg_type=WSMessageType.ORDER_UPDATE,
                            symbol=order.symbol,
                            data=order
                        )

            # 持倉更新
            elif topic == "position":
                positions = data.get("data", [])
                if positions:
                    account = self._parse_position_update(positions)
                    if account:
                        return WSMessage(
                            msg_type=WSMessageType.ACCOUNT_UPDATE,
                            data=account
                        )

            # 錢包更新
            elif topic == "wallet":
                wallet_data = data.get("data", [])
                if wallet_data:
                    account = self._parse_wallet_update(wallet_data)
                    if account:
                        return WSMessage(
                            msg_type=WSMessageType.ACCOUNT_UPDATE,
                            data=account
                        )

            return None

        except Exception as e:
            logger.error(f"[Bybit] 解析 WS 消息失敗: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 內部解析方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _parse_ticker(self, data: dict) -> Optional[TickerUpdate]:
        """
        解析 Bybit ticker 消息

        格式: {
            "symbol": "BTCUSDT",
            "lastPrice": "50000.00",
            "bid1Price": "49999.00",
            "ask1Price": "50001.00",
            "timestamp": "1234567890000"
        }
        """
        try:
            return TickerUpdate(
                symbol=self.convert_symbol_to_ccxt(data.get("symbol", "")),
                price=float(data.get("lastPrice", 0)),
                bid=float(data.get("bid1Price", 0)),
                ask=float(data.get("ask1Price", 0)),
                timestamp=float(data.get("timestamp", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_order_update(self, order_data: dict) -> Optional[OrderUpdate]:
        """
        解析 Bybit 訂單更新

        格式: {
            "symbol": "BTCUSDT",
            "orderId": "123456",
            "side": "Buy",
            "orderType": "Limit",
            "orderStatus": "Filled",
            "qty": "0.01",
            "cumExecQty": "0.01",
            "price": "50000",
            "avgPrice": "50000",
            "cumExecFee": "0.001",
            "reduceOnly": false,
            "positionIdx": 0
        }
        """
        try:
            # 轉換 position side
            pos_idx = order_data.get("positionIdx", 0)
            if pos_idx == 1:
                position_side = "LONG"
            elif pos_idx == 2:
                position_side = "SHORT"
            else:
                position_side = "BOTH"

            # 轉換 status
            status_map = {
                "New": "NEW",
                "PartiallyFilled": "PARTIALLY_FILLED",
                "Filled": "FILLED",
                "Cancelled": "CANCELED",
                "Rejected": "EXPIRED",
            }
            status = status_map.get(order_data.get("orderStatus", ""), "UNKNOWN")

            return OrderUpdate(
                symbol=order_data.get("symbol", ""),
                order_id=str(order_data.get("orderId", "")),
                side=order_data.get("side", "").upper(),
                position_side=position_side,
                status=status,
                order_type=order_data.get("orderType", "").upper(),
                quantity=float(order_data.get("qty", 0)),
                filled_quantity=float(order_data.get("cumExecQty", 0)),
                price=float(order_data.get("price", 0)),
                avg_price=float(order_data.get("avgPrice", 0)),
                realized_pnl=float(order_data.get("closedPnl", 0)),
                commission=float(order_data.get("cumExecFee", 0)),
                is_reduce_only=order_data.get("reduceOnly", False),
                timestamp=float(order_data.get("updatedTime", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_position_update(self, positions: list) -> Optional[AccountUpdate]:
        """
        解析 Bybit 持倉更新

        格式: [{
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "0.01",
            "avgPrice": "50000",
            "unrealisedPnl": "10",
            "leverage": "10"
        }]
        """
        try:
            result = []
            for pos in positions:
                size = float(pos.get("size", 0))
                if size == 0:
                    continue

                side = pos.get("side", "").upper()
                if side == "BUY":
                    position_side = "LONG"
                elif side == "SELL":
                    position_side = "SHORT"
                else:
                    continue

                result.append(PositionUpdate(
                    symbol=pos.get("symbol", ""),
                    position_side=position_side,
                    quantity=abs(size),
                    entry_price=float(pos.get("avgPrice", 0)),
                    unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                    leverage=int(pos.get("leverage", 1)),
                ))

            return AccountUpdate(
                positions=result,
                balances=[],
                timestamp=time.time(),
            )
        except Exception:
            return None

    def _parse_wallet_update(self, wallet_data: list) -> Optional[AccountUpdate]:
        """
        解析 Bybit 錢包更新

        格式: [{
            "accountType": "UNIFIED",
            "coin": [{
                "coin": "USDT",
                "walletBalance": "1000",
                "availableToWithdraw": "900"
            }]
        }]
        """
        try:
            balances = []
            for account in wallet_data:
                coins = account.get("coin", [])
                for coin in coins:
                    currency = coin.get("coin", "")
                    if currency not in ["USDC", "USDT"]:
                        continue

                    balances.append(BalanceUpdate(
                        currency=currency,
                        wallet_balance=float(coin.get("walletBalance", 0)),
                        available_balance=float(coin.get("availableToWithdraw", 0)),
                    ))

            return AccountUpdate(
                positions=[],
                balances=balances,
                timestamp=time.time(),
            )
        except Exception:
            return None
