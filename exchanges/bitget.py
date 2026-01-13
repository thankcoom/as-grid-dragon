# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Bitget Adapter
==============
Bitget 交易所適配器實作

WebSocket 訊息格式:
- Ticker: {"action": "snapshot", "arg": {"instType": "USDT-FUTURES"}, "data": [...]}
- Order: {"action": "snapshot", "arg": {"channel": "orders"}, "data": [...]}
- Position: {"action": "snapshot", "arg": {"channel": "positions"}, "data": [...]}
"""

import json
import logging
import time
import hmac
import base64
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

# Bitget WebSocket URLs
BITGET_WS_MAINNET = "wss://ws.bitget.com/v2/ws/private"
BITGET_WS_PUBLIC_MAINNET = "wss://ws.bitget.com/v2/ws/public"
BITGET_WS_TESTNET = "wss://ws.bitget.com/v2/ws/private"  # Bitget 測試網相同 URL

# Bitget 心跳間隔 (30 秒)
BITGET_PING_INTERVAL = 30


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Bitget Adapter                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BitgetAdapter(ExchangeAdapter):
    """
    Bitget 期貨交易所適配器

    Features:
        - CCXT bitget 封裝
        - WebSocket 訂閱 (ticker, order, position)
        - 標準化 WebSocket 消息解析

    注意: Bitget 需要額外的 password (API passphrase)
    """

    def __init__(self):
        super().__init__()
        self._testnet = False
        self._api_key: str = ""
        self._api_secret: str = ""
        self._password: str = ""  # Bitget 特有

    # ═══════════════════════════════════════════════════════════════════════════
    # 基本資訊
    # ═══════════════════════════════════════════════════════════════════════════

    def get_exchange_name(self) -> str:
        return "bitget"

    def get_display_name(self) -> str:
        return "Bitget"

    # ═══════════════════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════════════════

    def init_exchange(self, api_key: str, api_secret: str,
                      testnet: bool = False, password: str = "") -> None:
        """
        初始化 Bitget CCXT 實例

        Args:
            api_key: API Key
            api_secret: API Secret
            testnet: 是否使用測試網
            password: API Passphrase (Bitget 必需)
        """
        self._testnet = testnet
        self._api_key = api_key
        self._api_secret = api_secret
        self._password = password

        options = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,  # Bitget 特有
            "options": {"defaultType": "swap"}  # USDT 永續
        }

        if testnet:
            options["sandbox"] = True

        self.exchange = ccxt.bitget(options)
        self.exchange.options["defaultType"] = "swap"

        logger.info(f"[Bitget] 交易所初始化完成 (testnet={testnet})")

    def load_markets(self) -> None:
        """載入市場資訊"""
        if not self.exchange:
            raise RuntimeError("請先呼叫 init_exchange()")

        self.exchange.load_markets(reload=False)
        self._markets_loaded = True
        logger.info(f"[Bitget] 已載入 {len(self.exchange.markets)} 個市場")

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
                min_notional=5.0,  # Bitget 最小名義價值
                tick_size=price_prec,
                step_size=amount_prec,
            )
        except Exception as e:
            logger.error(f"[Bitget] 獲取 {symbol} 精度失敗: {e}")
            return PrecisionInfo(
                price_precision=4,
                amount_precision=0,
                min_quantity=1,
                min_notional=5.0,
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

        logger.warning(f"[Bitget] 無法轉換符號: {raw_symbol}")
        return raw_symbol

    def convert_symbol_to_ws(self, raw_symbol: str) -> str:
        """
        將原始交易對符號轉換為 WebSocket 訂閱格式

        Bitget 使用大寫符號
        Examples:
            XRP/USDT:USDT -> XRPUSDT
            XRPUSDT -> XRPUSDT
        """
        # 處理 CCXT 格式 (移除 :USDT 後綴)
        if ":" in raw_symbol:
            raw_symbol = raw_symbol.split(":")[0]

        ws_sym = raw_symbol.replace("/", "").replace(":", "")
        return ws_sym.upper()

    # ═══════════════════════════════════════════════════════════════════════════
    # REST API - 帳戶
    # ═══════════════════════════════════════════════════════════════════════════

    def fetch_balance(self) -> Dict[str, BalanceUpdate]:
        """獲取帳戶餘額"""
        result = {}

        try:
            balance = self.exchange.fetch_balance({"type": "swap"})

            for currency in ["USDC", "USDT"]:
                if currency in balance:
                    info = balance[currency]
                    result[currency] = BalanceUpdate(
                        currency=currency,
                        wallet_balance=float(info.get("total", 0) or 0),
                        available_balance=float(info.get("free", 0) or 0),
                    )
        except Exception as e:
            logger.error(f"[Bitget] 獲取餘額失敗: {e}")

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
            logger.error(f"[Bitget] 獲取持倉失敗: {e}")

        return result

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """設定槓桿"""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"[Bitget] {symbol} 槓桿設為 {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"[Bitget] 設置 {symbol} 槓桿失敗: {e}")
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

        # Bitget 使用 holdSide
        if position_side == "LONG":
            params["holdSide"] = "long"
        elif position_side == "SHORT":
            params["holdSide"] = "short"

        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=amount,
            price=price,
            params=params
        )

        logger.info(f"[Bitget] 限價單: {symbol} {side} {amount}@{price}")
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
            params["holdSide"] = "long"
        elif position_side == "SHORT":
            params["holdSide"] = "short"

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=amount,
            params=params
        )

        logger.info(f"[Bitget] 市價單: {symbol} {side} {amount}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消訂單"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"[Bitget] 取消訂單失敗: {e}")
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """獲取未成交訂單"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"[Bitget] 獲取掛單失敗: {e}")
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
            logger.error(f"[Bitget] 獲取資金費率失敗: {e}")
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # WebSocket
    # ═══════════════════════════════════════════════════════════════════════════

    def get_websocket_url(self) -> str:
        """獲取 WebSocket 基礎 URL (私有頻道)"""
        return BITGET_WS_MAINNET

    def get_public_websocket_url(self) -> str:
        """獲取公共 WebSocket URL"""
        return BITGET_WS_PUBLIC_MAINNET

    async def start_user_stream(self) -> Optional[str]:
        """
        Bitget 使用 API 簽名認證
        返回認證參數 JSON 字串
        """
        if not self._api_key or not self._api_secret:
            logger.error("[Bitget] 缺少 API 憑證")
            return None

        # 生成認證參數
        timestamp = str(int(time.time()))
        signature = self._generate_signature(timestamp)

        login_params = {
            "op": "login",
            "args": [{
                "apiKey": self._api_key,
                "passphrase": self._password,
                "timestamp": timestamp,
                "sign": signature
            }]
        }

        logger.info("[Bitget] 已準備 WebSocket 認證參數")
        return json.dumps(login_params)

    def _generate_signature(self, timestamp: str) -> str:
        """生成 WebSocket 認證簽名"""
        message = f"{timestamp}GET/user/verify"
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode("utf-8")

    async def keepalive_user_stream(self) -> None:
        """Bitget 需要定期發送 ping"""
        pass

    def get_keepalive_interval(self) -> int:
        """獲取 keepalive 間隔 (秒)"""
        return BITGET_PING_INTERVAL

    def build_stream_url(
        self,
        symbols: List[str],
        user_stream_key: Optional[str] = None
    ) -> str:
        """建構完整的 WebSocket 訂閱 URL"""
        return self.get_websocket_url()

    def get_subscription_message(self, symbols: List[str]) -> str:
        """
        生成訂閱消息

        Args:
            symbols: 要訂閱的交易對列表

        Returns:
            JSON 訂閱消息
        """
        args = []

        # 訂閱訂單
        args.append({
            "instType": "USDT-FUTURES",
            "channel": "orders",
            "instId": "default"
        })

        # 訂閱持倉
        args.append({
            "instType": "USDT-FUTURES",
            "channel": "positions",
            "instId": "default"
        })

        # 訂閱帳戶
        args.append({
            "instType": "USDT-FUTURES",
            "channel": "account",
            "coin": "default"
        })

        # 訂閱 ticker (公共頻道)
        for symbol in symbols:
            ws_sym = self.convert_symbol_to_ws(symbol)
            print(f"[DEBUG] Bitget WS Subscription: symbol={symbol} -> ws_symbol={ws_sym}")
            args.append({
                "instType": "USDT-FUTURES",
                "channel": "ticker",
                "instId": ws_sym
            })

        subscribe_msg = {
            "op": "subscribe",
            "args": args
        }

        return json.dumps(subscribe_msg)

    def parse_ws_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        解析 WebSocket 原始消息

        Bitget 消息格式:
            {"action": "snapshot", "arg": {"channel": "ticker"}, "data": [...]}
        """
        try:
            data = json.loads(raw_message)

            # 處理系統消息
            if "event" in data:
                event = data.get("event")
                if event in ["subscribe", "login"]:
                    logger.debug(f"[Bitget] 系統消息: {event}")
                    return None

            # 處理 pong
            if data.get("op") == "pong":
                return None

            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            payload = data.get("data", [])

            if not payload:
                return None

            # Ticker 更新
            if channel == "ticker":
                ticker = self._parse_ticker(payload[0])
                if ticker:
                    return WSMessage(
                        msg_type=WSMessageType.TICKER,
                        symbol=ticker.symbol,
                        data=ticker
                    )

            # 訂單更新
            elif channel == "orders":
                order = self._parse_order_update(payload[0])
                if order:
                    return WSMessage(
                        msg_type=WSMessageType.ORDER_UPDATE,
                        symbol=order.symbol,
                        data=order
                    )

            # 持倉更新
            elif channel == "positions":
                account = self._parse_position_update(payload)
                if account:
                    return WSMessage(
                        msg_type=WSMessageType.ACCOUNT_UPDATE,
                        data=account
                    )

            # 帳戶更新
            elif channel == "account":
                account = self._parse_account_data(payload)
                if account:
                    return WSMessage(
                        msg_type=WSMessageType.ACCOUNT_UPDATE,
                        data=account
                    )

            return None

        except Exception as e:
            logger.error(f"[Bitget] 解析 WS 消息失敗: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # 內部解析方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _parse_ticker(self, data: dict) -> Optional[TickerUpdate]:
        """
        解析 Bitget ticker 消息

        格式: {
            "instId": "BTCUSDT",
            "last": "50000.00",
            "bestBid": "49999.00",
            "bestAsk": "50001.00",
            "ts": "1234567890000"
        }
        """
        try:
            return TickerUpdate(
                symbol=self.convert_symbol_to_ccxt(data.get("instId", "")),
                price=float(data.get("last", 0)),
                bid=float(data.get("bestBid", 0)),
                ask=float(data.get("bestAsk", 0)),
                timestamp=float(data.get("ts", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_order_update(self, order_data: dict) -> Optional[OrderUpdate]:
        """
        解析 Bitget 訂單更新

        格式: {
            "instId": "BTCUSDT",
            "ordId": "123456",
            "side": "buy",
            "ordType": "limit",
            "status": "filled",
            "sz": "0.01",
            "fillSz": "0.01",
            "px": "50000",
            "avgPx": "50000",
            "fee": "-0.001",
            "reduceOnly": "false",
            "posSide": "long"
        }
        """
        try:
            # 轉換 position side
            pos_side = order_data.get("posSide", "").upper()
            if pos_side not in ["LONG", "SHORT"]:
                pos_side = "BOTH"

            # 轉換 status
            status_map = {
                "live": "NEW",
                "partially_filled": "PARTIALLY_FILLED",
                "filled": "FILLED",
                "cancelled": "CANCELED",
                "canceled": "CANCELED",
            }
            raw_status = order_data.get("status", "").lower()
            status = status_map.get(raw_status, "UNKNOWN")

            return OrderUpdate(
                symbol=order_data.get("instId", ""),
                order_id=str(order_data.get("ordId", "")),
                side=order_data.get("side", "").upper(),
                position_side=pos_side,
                status=status,
                order_type=order_data.get("ordType", "").upper(),
                quantity=float(order_data.get("sz", 0)),
                filled_quantity=float(order_data.get("fillSz", 0)),
                price=float(order_data.get("px", 0)),
                avg_price=float(order_data.get("avgPx", 0) or 0),
                realized_pnl=float(order_data.get("pnl", 0) or 0),
                commission=abs(float(order_data.get("fee", 0) or 0)),
                is_reduce_only=str(order_data.get("reduceOnly", "false")).lower() == "true",
                timestamp=float(order_data.get("uTime", 0)) / 1000,
            )
        except Exception:
            return None

    def _parse_position_update(self, positions: list) -> Optional[AccountUpdate]:
        """
        解析 Bitget 持倉更新

        格式: [{
            "instId": "BTCUSDT",
            "holdSide": "long",
            "total": "0.01",
            "avgPx": "50000",
            "upl": "10",
            "lever": "10"
        }]
        """
        try:
            result = []
            for pos in positions:
                total = float(pos.get("total", 0))
                if total == 0:
                    continue

                hold_side = pos.get("holdSide", "").upper()
                if hold_side not in ["LONG", "SHORT"]:
                    continue

                result.append(PositionUpdate(
                    symbol=pos.get("instId", ""),
                    position_side=hold_side,
                    quantity=abs(total),
                    entry_price=float(pos.get("avgPx", 0)),
                    unrealized_pnl=float(pos.get("upl", 0)),
                    leverage=int(pos.get("lever", 1) or 1),
                ))

            return AccountUpdate(
                positions=result,
                balances=[],
                timestamp=time.time(),
            )
        except Exception:
            return None

    def _parse_account_data(self, account_data: list) -> Optional[AccountUpdate]:
        """
        解析 Bitget 帳戶更新

        格式: [{
            "coin": "USDT",
            "available": "1000",
            "equity": "1100"
        }]
        """
        try:
            balances = []
            for acc in account_data:
                currency = acc.get("coin", "").upper()
                if currency not in ["USDC", "USDT"]:
                    continue

                balances.append(BalanceUpdate(
                    currency=currency,
                    wallet_balance=float(acc.get("equity", 0)),
                    available_balance=float(acc.get("available", 0)),
                ))

            return AccountUpdate(
                positions=[],
                balances=balances,
                timestamp=time.time(),
            )
        except Exception:
            return None
