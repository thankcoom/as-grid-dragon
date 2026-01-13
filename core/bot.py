# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
MaxGridBot - 主交易機器人
========================
WebSocket 連接、交易執行、狀態管理

架構:
    GridBot 使用 ExchangeAdapter 抽象層與交易所互動，
    支援任何 CCXT 兼容的交易所。
"""

import json
import time
import asyncio
import logging
import ssl
import certifi
from datetime import datetime
from typing import Dict, Optional

import websockets

from .strategy import GridStrategy
from config.models import GlobalConfig, GlobalState, SymbolState
from utils import safe_float
from indicators.bandit import UCBBanditOptimizer
from indicators.leading import LeadingIndicatorManager
from indicators.funding import FundingRateManager, GLFTController
from indicators.dgt import DGTBoundaryManager, DynamicGridManager

# 多交易所支援
from exchanges.bybit import BybitAdapter
from exchanges.gate import GateAdapter
from exchanges import get_adapter, ExchangeAdapter
from exchanges.base import WSMessageType

from utils import normalize_symbol

logger = logging.getLogger("as_grid_max")





class MaxGridBot:
    """MAX 網格交易機器人"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.state = GlobalState()

        # === 多交易所支援 ===
        exchange_type = getattr(config, 'exchange_type', 'binance')
        self.adapter: ExchangeAdapter = get_adapter(exchange_type)

        # === 舊版兼容: 保留 self.exchange 引用 ===
        self.exchange = None  # 將在 _init_exchange() 中設定

        self.ws = None
        self._stop_event = asyncio.Event()
        self.precision_info: Dict[str, Dict] = {}
        self.bandit_optimizer = UCBBanditOptimizer(config.bandit)
        self.leading_indicator = LeadingIndicatorManager(config.leading_indicator)
        self.funding_manager: Optional[FundingRateManager] = None
        self.glft_controller = GLFTController()
        self.dgt_manager = DGTBoundaryManager(config.dgt)
        self.dynamic_grid = DynamicGridManager()
        self.last_grid_time: Dict[str, float] = {}
        self.grid_interval = 0.5
        # User Data Stream
        self.listen_key: Optional[str] = None
        self.listen_key_time: float = 0

    def _init_exchange(self):
        """初始化交易所 (使用 Adapter 抽象層)"""
        # 使用 Adapter 初始化
        testnet = getattr(self.config, 'testnet', False)
        password = getattr(self.config, 'api_password', '')  # Bitget 需要 passphrase
        self.adapter.init_exchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=testnet,
            password=password  # 傳遞 Bitget passphrase
        )
        self.adapter.load_markets()

        # === 自動遷移舊配置 Key (Raw -> CCXT) ===
        migrated = False
        new_symbols = {}
        for key, sym_cfg in list(self.config.symbols.items()):
            # 使用 normalize_symbol 重新解析，確保資料正確
            _, real_ccxt, _, _ = normalize_symbol(key)
            
            # 如果解析失敗，嘗試用 ccxt_symbol 解析
            if not real_ccxt:
                _, real_ccxt, _, _ = normalize_symbol(sym_cfg.ccxt_symbol)

            # 遷移判定: (Key 不正確) 或 (Config 內的 ccxt_symbol 不正確)
            if real_ccxt and (key != real_ccxt or sym_cfg.ccxt_symbol != real_ccxt):
                logger.warning(f"[Bot] 遷移配置: {key} -> {real_ccxt}")
                
                # 修正 Config 對象內的數據
                sym_cfg.ccxt_symbol = real_ccxt
                
                # 使用正確的 Key 存儲
                new_symbols[real_ccxt] = sym_cfg
                migrated = True
            else:
                # 已經正確，直接保留
                new_symbols[key] = sym_cfg
        
        if migrated:
            self.config.symbols = new_symbols
            logger.info(f"[Bot] 配置遷移完成，新 Key: {list(new_symbols.keys())}")

        # === 初始化 State Symbols (關鍵: 確保 State 中有 Key 才能接收更新) ===
        for sym_cfg in self.config.symbols.values():
            if sym_cfg.enabled:
                if sym_cfg.ccxt_symbol not in self.state.symbols:
                    self.state.symbols[sym_cfg.ccxt_symbol] = SymbolState(symbol=sym_cfg.ccxt_symbol)
                # 確保 Config 引用正確 (選填)
                pass

        # === 舊版兼容: 保留 self.exchange 引用 ===
        self.exchange = self.adapter.exchange

        # 載入精度資訊
        for sym_cfg in self.config.symbols.values():
            if not sym_cfg.enabled:
                continue
            try:
                precision = self.adapter.get_precision(sym_cfg.ccxt_symbol)
                self.precision_info[sym_cfg.ccxt_symbol] = {
                    "price": int(precision.price_precision),
                    "amount": int(precision.amount_precision),
                    "min_notional": precision.min_notional
                }
            except Exception as e:
                logger.error(f"獲取 {sym_cfg.ccxt_symbol} 精度失敗: {e}")

        # 檢查並啟用對沖模式 (與 Terminal 版一致)
        self._check_hedge_mode()

        self.funding_manager = FundingRateManager(self.exchange)
        logger.info(f"[Bot] {self.adapter.get_display_name()} 初始化完成，{len(self.precision_info)} 個交易對")

    def _check_hedge_mode(self):
        """
        檢查並啟用對沖模式 (與 Terminal 版一致)
        
        Binance 需要啟用 Hedge Mode (雙向持倉) 才能同時持有多空倉位
        """
        for sym_config in self.config.symbols.values():
            if sym_config.enabled:
                try:
                    mode = self.exchange.fetch_position_mode(symbol=sym_config.ccxt_symbol)
                    if not mode.get('hedged', False):
                        print(f"[Bot] {sym_config.symbol} 啟用對沖模式...")
                        self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                        print(f"[Bot] 對沖模式已啟用")
                        break
                except Exception as e:
                    logger.warning(f"[Bot] 檢查對沖模式失敗: {e}")

    def _init_state(self):
        for sym_cfg in self.config.symbols.values():
            if sym_cfg.enabled:
                self.state.symbols[sym_cfg.ccxt_symbol] = SymbolState(symbol=sym_cfg.symbol)
        self.state.start_time = datetime.now()
        self.state.running = True

    async def _get_listen_key(self) -> Optional[str]:
        """獲取 User Data Stream Key (使用 Adapter)"""
        self.listen_key = await self.adapter.start_user_stream()
        if self.listen_key:
            self.listen_key_time = time.time()
        return self.listen_key

    async def _keepalive_listen_key(self):
        """定期延長 User Data Stream (使用 Adapter)"""
        interval = self.adapter.get_keepalive_interval()
        if interval <= 0:
            return  # 此交易所不需要 keepalive

        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            try:
                await self.adapter.keepalive_user_stream()
                self.listen_key_time = time.time()
            except Exception as e:
                logger.error(f"[Bot] Keepalive 失敗: {e}")
                await self._get_listen_key()

    async def _sync_positions(self):
        try:
            # 使用 Adapter 獲取標準化的持倉資料
            positions = self.adapter.fetch_positions()
            for pos in positions:
                # 使用標準化 symbol 匹配
                normalized_sym = normalize_symbol(pos.symbol)[0]
                ccxt_symbol = None
                for cfg in self.config.symbols.values():
                    cfg_normalized = normalize_symbol(cfg.symbol)[0]
                    if cfg_normalized == normalized_sym:
                        ccxt_symbol = cfg.ccxt_symbol
                        break
                if not ccxt_symbol or ccxt_symbol not in self.state.symbols:
                    continue

                side = pos.position_side.lower()  # LONG 或 SHORT
                if side == "long":
                    self.state.symbols[ccxt_symbol].long_position = pos.quantity
                elif side == "short":
                    self.state.symbols[ccxt_symbol].short_position = pos.quantity
                self.state.symbols[ccxt_symbol].unrealized_pnl = pos.unrealized_pnl

            # 使用 Adapter 獲取標準化的餘額資料
            balances = self.adapter.fetch_balance()
            for currency in ["USDC", "USDT"]:
                if currency in balances:
                    bal = balances[currency]
                    acc = self.state.get_account(currency)
                    acc.wallet_balance = bal.wallet_balance
                    acc.available_balance = bal.available_balance
                    # 計算保證金使用
                    acc.margin_used = acc.wallet_balance - acc.available_balance
            self.state.update_totals()
        except Exception as e:
            logger.error(f"同步倉位失敗: {e}")

    async def _sync_orders(self):
        """
        同步掛單狀態 (與 Terminal 版一致)
        
        追蹤每個交易對的掛單數量:
        - buy_long_orders: 多頭補倉單數量
        - sell_long_orders: 多頭止盈單數量
        - buy_short_orders: 空頭止盈單數量
        - sell_short_orders: 空頭補倉單數量
        """
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            ccxt_symbol = sym_config.ccxt_symbol

            try:
                orders = self.adapter.fetch_open_orders(ccxt_symbol)
                state = self.state.symbols.get(ccxt_symbol)
                if not state:
                    continue

                # 重置掛單計數
                state.buy_long_orders = 0
                state.sell_long_orders = 0
                state.buy_short_orders = 0
                state.sell_short_orders = 0

                for order in orders:
                    qty = abs(float(order.get('amount', 0) or order.get('info', {}).get('origQty', 0)))
                    side = order.get('side', '').lower()
                    pos_side = order.get('info', {}).get('positionSide', 'BOTH').upper()

                    if side == 'buy' and pos_side == 'LONG':
                        state.buy_long_orders += qty
                    elif side == 'sell' and pos_side == 'LONG':
                        state.sell_long_orders += qty
                    elif side == 'buy' and pos_side == 'SHORT':
                        state.buy_short_orders += qty
                    elif side == 'sell' and pos_side == 'SHORT':
                        state.sell_short_orders += qty

            except Exception as e:
                logger.error(f"同步 {ccxt_symbol} 掛單失敗: {e}")

    async def _sync_funding_rates(self):
        """
        同步所有交易對的 Funding Rate (與 Terminal 版一致)
        
        用於:
        1. Funding Rate 偏向機制
        2. 調整多空持倉偏好
        """
        if not self.funding_manager:
            return
        
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            
            try:
                rate = self.funding_manager.update_funding_rate(sym_config.ccxt_symbol)
                sym_state = self.state.symbols.get(sym_config.ccxt_symbol)
                if sym_state:
                    sym_state.current_funding_rate = rate
            except Exception as e:
                logger.debug(f"同步 {sym_config.symbol} Funding Rate 失敗: {e}")

    async def run(self):
        try:
            print("[Bot] 開始初始化交易所...")
            self._init_exchange()
            print("[Bot] 交易所初始化完成")

            self._init_state()
            print("[Bot] 狀態初始化完成，state.running = True")

            await self._sync_positions()
            print("[Bot] 倉位同步完成")

            for cfg in self.config.symbols.values():
                if cfg.enabled:
                    try:
                        self.adapter.set_leverage(cfg.ccxt_symbol, cfg.leverage)
                    except Exception as e:
                        print(f"[Bot] 設置 {cfg.symbol} 槓桿失敗: {e}")

            # 掛初始網格
            await self._place_initial_grids()
            print("[Bot] 啟動 WebSocket...")
            await self._websocket_loop()
        except Exception as e:
            print(f"[Bot] run() 執行失敗: {type(e).__name__}: {e}")
            raise  # 重新拋出讓外層捕獲

    async def _place_initial_grids(self):
        """啟動時掛初始網格"""
        for cfg in self.config.symbols.values():
            if cfg.enabled:
                await self._place_grid(cfg)
                logger.info(f"[Bot] {cfg.symbol} 初始網格已掛")
                await asyncio.sleep(0.2)  # 避免 API 限流

    async def stop(self):
        self._stop_event.set()
        self.state.running = False
        if self.ws:
            await self.ws.close()

    def reload_config(self, new_config: GlobalConfig):
        """
        重新載入配置並套用到運行中的 Bot (與 Terminal 版一致)
        
        可動態更新:
        - symbols 參數 (間距、數量等)
        - risk 參數
        - bandit 參數
        - leading_indicator 參數
        """
        # 保留關鍵運行狀態
        old_api_key = self.config.api_key
        old_api_secret = self.config.api_secret
        
        # 更新配置
        self.config = new_config
        
        # 確保 API 密鑰不變 (運行中不可更改)
        self.config.api_key = old_api_key
        self.config.api_secret = old_api_secret
        
        # 更新 Bandit 配置
        if hasattr(self.bandit_optimizer, 'config'):
            self.bandit_optimizer.config = new_config.bandit
        
        # 更新領先指標配置
        if hasattr(self.leading_indicator, 'config'):
            self.leading_indicator.config = new_config.leading_indicator
        
        logger.info("[Bot] 配置已重新載入")

    async def _websocket_loop(self):
        """WebSocket 主循環 (使用 Adapter 構建 URL) - 參考 as_terminal_max.py 邏輯優化"""
        # 收集啟用的交易對
        symbols = []
        for sym_cfg in self.config.symbols.values():
            if sym_cfg.enabled:
                symbols.append(sym_cfg.symbol)

        if not symbols:
            logger.error("[Bot] 沒有啟用的交易對")
            return

        # 獲取 Listen Key
        await self._get_listen_key()

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # 啟動 keepalive
        asyncio.create_task(self._keepalive_listen_key())

        while not self._stop_event.is_set():
            try:
                # 使用 Adapter 構建 URL
                url = self.adapter.build_stream_url(symbols, self.listen_key)

                print(f"[Bot] 正在連接 WebSocket: {url[:80]}...")
                # 參考 as_terminal_max.py: 移除 ping_interval, 使用手動 ping
                async with websockets.connect(url, ssl=ssl_context) as ws:
                    self.ws = ws
                    self.state.connected = True
                    print(f"[Bot] WebSocket 已連接 ({self.adapter.get_display_name()})")

                    # 發送認證訊息 (Bitget 需要)
                    if self.listen_key and self.listen_key.startswith("{"):
                        print(f"[Bot] 發送認證訊息...")
                        await ws.send(self.listen_key)
                        await asyncio.sleep(1.0)  # 等待認證成功

                    # 對於 Binance，訂閱請求已經在 URL 中完成 (Combined Stream)
                    # 只有其他交易所需要額外發送訂閱消息
                    if self.adapter.get_exchange_name() != 'binance':
                        # 檢查 adapter 是否有 get_subscription_message 方法
                        if hasattr(self.adapter, 'get_subscription_message'):
                            sub_msg = self.adapter.get_subscription_message(symbols)
                            if sub_msg:
                                print(f"[Bot] 發送訂閱請求: {str(sub_msg)[:100]}...")
                                # 修正: 如果 sub_msg 已經是字串 (JSON)，直接發送，避免 double encoding
                                if isinstance(sub_msg, str):
                                    await ws.send(sub_msg)
                                else:
                                    await ws.send(json.dumps(sub_msg))

                    # 啟動同步循環
                    sync_task = asyncio.create_task(self._sync_loop())
                    
                    try:
                        while not self._stop_event.is_set():
                            try:
                                # 參考 as_terminal_max.py: 使用 wait_for + 手動 ping
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                await self._handle_message(msg)
                            except asyncio.TimeoutError:
                                # 超時發送 ping
                                await ws.ping()
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("[Bot] WebSocket 連接關閉")
                    finally:
                        sync_task.cancel()

            except Exception as e:
                logger.error(f"[Bot] WebSocket 錯誤: {e}")
                
            self.state.connected = False
            if not self._stop_event.is_set():
                print("[Bot] WebSocket 斷開，5秒後重連...")
                await asyncio.sleep(5)

    async def _sync_loop(self):
        """定期同步循環 (與 Terminal 版 sync_all 一致)"""
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.sync_interval)
            await self._sync_positions()
            await self._sync_orders()
            await self._sync_funding_rates()  # 同步 Funding Rate
            # 風控檢查
            await self._risk_monitor_loop()
            # 減倉檢查
            await self._check_and_reduce_positions()

    async def _handle_message(self, message: str):
        """處理 WebSocket 消息 (使用 Adapter 解析)"""
        try:
            ws_msg = self.adapter.parse_ws_message(message)
            if not ws_msg:
                # 添加調試日誌，檢查解析失敗的消息
                logger.debug(f"[WebSocket] 無法解析消息: {message[:200]}")
                return

            if ws_msg.msg_type == WSMessageType.TICKER:
                logger.debug(f"[WebSocket] 收到Ticker消息: {ws_msg.symbol}, 價格={getattr(ws_msg.data, 'price', 'N/A')}")
                await self._handle_ticker_update(ws_msg.symbol, ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ORDER_UPDATE:
                logger.debug(f"[WebSocket] 收到訂單消息: {ws_msg.symbol}, 狀態={getattr(ws_msg.data, 'status', 'N/A')}")
                await self._handle_order_update(ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ACCOUNT_UPDATE:
                logger.debug(f"[WebSocket] 收到账戶消息: 更新持倉和餘額")
                await self._handle_account_update(ws_msg.data)

        except Exception as e:
            logger.error(f"處理訊息錯誤: {e}")
            logger.debug(f"錯誤消息內容: {message[:500]}")

    async def _handle_order_update(self, order_update):
        """
        處理訂單更新 (標準化 OrderUpdate 格式)

        Args:
            order_update: exchanges.base.OrderUpdate 實例
        """
        # 只處理完全成交
        if order_update.status != "FILLED":
            return

        symbol = order_update.symbol  # 原始交易對 (如 XRPUSDC 或 XRP_USDT)
        side = order_update.side      # BUY 或 SELL
        qty = order_update.filled_quantity
        price = order_update.avg_price
        realized_pnl = order_update.realized_pnl
        is_reduce = order_update.is_reduce_only

        logger.info(f"[Bot] 訂單成交: {symbol} {side} {qty}@{price} PnL={realized_pnl:.4f}")

        # 找到對應的 ccxt_symbol (處理不同交易所的 symbol 格式)
        ccxt_symbol = None
        normalized_symbol = normalize_symbol(symbol)[0]
        for cfg in self.config.symbols.values():
            cfg_normalized = normalize_symbol(cfg.symbol)[0]
            if cfg_normalized == normalized_symbol:
                ccxt_symbol = cfg.ccxt_symbol
                break

        if not ccxt_symbol:
            return

        sym_state = self.state.symbols.get(ccxt_symbol)
        if not sym_state:
            return

        # 更新交易統計
        sym_state.total_trades += 1
        sym_state.total_profit += realized_pnl
        self.state.total_trades += 1
        self.state.total_profit += realized_pnl

        # 記錄最近交易
        sym_state.recent_trades.append({
            "time": datetime.now().isoformat(),
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": realized_pnl
        })

        # === 關鍵: 呼叫 Bandit 記錄交易 ===
        if self.config.bandit.enabled and is_reduce and realized_pnl != 0:
            trade_side = "long" if side == "SELL" else "short"  # 減倉方向相反
            self.bandit_optimizer.record_trade(realized_pnl, trade_side)
            logger.debug(f"[Bandit] 記錄交易: {trade_side} pnl={realized_pnl:.4f}")

        # === 關鍵: 呼叫領先指標記錄 ===
        if self.config.leading_indicator.enabled:
            trade_side = "buy" if side == "BUY" else "sell"
            self.leading_indicator.record_trade(ccxt_symbol, price, qty, trade_side)

    async def _handle_account_update(self, account_update):
        """
        處理帳戶更新 (標準化 AccountUpdate 格式)

        Args:
            account_update: exchanges.base.AccountUpdate 實例
        """
        # 更新持倉
        for pos in account_update.positions:
            # 找到對應的 ccxt_symbol (處理不同交易所的 symbol 格式)
            ccxt_symbol = None
            normalized_symbol = pos.symbol.upper().replace("_", "").replace("/", "").replace(":", "")
            for cfg in self.config.symbols.values():
                cfg_normalized = normalize_symbol(cfg.symbol)[0]
                if cfg_normalized == normalized_symbol:
                    ccxt_symbol = cfg.ccxt_symbol
                    break
            if not ccxt_symbol or ccxt_symbol not in self.state.symbols:
                continue

            sym_state = self.state.symbols[ccxt_symbol]
            side = pos.position_side.lower()  # LONG 或 SHORT

            if side == "long":
                sym_state.long_position = pos.quantity
            else:
                sym_state.short_position = pos.quantity
            sym_state.unrealized_pnl = pos.unrealized_pnl

        # 更新餘額
        for bal in account_update.balances:
            if bal.currency in ["USDC", "USDT"]:
                acc = self.state.get_account(bal.currency)
                acc.wallet_balance = bal.wallet_balance
                acc.available_balance = bal.available_balance
                acc.margin_used = acc.wallet_balance - acc.available_balance

        self.state.update_totals()

    async def _handle_ticker_update(self, raw_symbol: str, ticker_update):
        """
        處理 Ticker 更新 (標準化 TickerUpdate 格式)

        Args:
            raw_symbol: 原始交易對符號 (如 XRPUSDC 或 XRP_USDT)
            ticker_update: exchanges.base.TickerUpdate 實例
        """
        # 改進符號匹配邏輯 - 先嘗試直接匹配，再標準化匹配
        matched = False
        matched_cfg = None
        
        # 1. 首先嘗試原始符號直接匹配 (對應 ccxt_symbol)
        for cfg in self.config.symbols.values():
            if not cfg.enabled:
                continue
            # 檢查原始符號是否與 ccxt_symbol 匹配
            if raw_symbol == cfg.ccxt_symbol and cfg.ccxt_symbol in self.state.symbols:
                matched_cfg = cfg
                matched = True
                break
        
        # 2. 如果沒有直接匹配，嘗試標準化匹配
        if not matched:
            normalized_raw = normalize_symbol(raw_symbol)[0]
            for cfg in self.config.symbols.values():
                if not cfg.enabled:
                    continue
                cfg_normalized = normalize_symbol(cfg.symbol)[0]
                
                if cfg_normalized == normalized_raw and cfg.ccxt_symbol in self.state.symbols:
                    matched_cfg = cfg
                    matched = True
                    break
        
        # 3. 如果還是沒有匹配，嘗試適配器的符號轉換
        if not matched:
            for cfg in self.config.symbols.values():
                if not cfg.enabled:
                    continue
                # 使用適配器轉換原始符號到WS格式，再反向匹配
                try:
                    ws_symbol = self.adapter.convert_symbol_to_ws(cfg.symbol)
                    if raw_symbol.upper() == ws_symbol.upper():
                        matched_cfg = cfg
                        matched = True
                        break
                    # 或者嘗試將WS格式轉換回CCXT格式
                    ccxt_from_ws = self.adapter.convert_symbol_to_ccxt(raw_symbol)
                    if ccxt_from_ws == cfg.ccxt_symbol:
                        matched_cfg = cfg
                        matched = True
                        break
                except Exception:
                    continue
        
        if matched and matched_cfg:
            sym_state = self.state.symbols[matched_cfg.ccxt_symbol]
            
            # 更新價格信息
            old_price = sym_state.latest_price
            sym_state.latest_price = ticker_update.price
            sym_state.best_bid = ticker_update.bid
            sym_state.best_ask = ticker_update.ask

            # 添加調試日誌
            if old_price != sym_state.latest_price:
                logger.debug(f"[Ticker] {matched_cfg.symbol} 價格更新: {old_price} -> {sym_state.latest_price}")

            # 更新領先指標
            self.leading_indicator.update_spread(matched_cfg.ccxt_symbol, sym_state.best_bid, sym_state.best_ask)
            self.dynamic_grid.update_price(matched_cfg.ccxt_symbol, sym_state.latest_price)

            if self.config.bandit.contextual_enabled:
                self.bandit_optimizer.update_price(sym_state.latest_price)

            # 總是嘗試執行網格交易，而不僅僅是間隔控制
            # 這樣可以確保價格變化時立即響應
            await self._place_grid(matched_cfg)
            
            # 如果價格變動較大，也更新時間戳以避免過於頻繁的網格操作
            now = time.time()
            last = self.last_grid_time.get(matched_cfg.ccxt_symbol, 0)
            if now - last >= self.grid_interval:
                self.last_grid_time[matched_cfg.ccxt_symbol] = now


    async def _place_grid(self, cfg):
        """
        掛出網格訂單 (與 Terminal 版一致)
        
        邏輯:
        - 持倉為 0 時: 在 best_bid/best_ask 下單開倉
        - 有持倉時: 下止盈單和補倉單
        """
        ccxt_sym = cfg.ccxt_symbol
        sym_state = self.state.symbols.get(ccxt_sym)
        if not sym_state:
            logger.warning(f"[Bot] 無法找到 {ccxt_sym} 的狀態")
            return
        
        # 使用當前價格，如果價格未更新，則跳過本次網格操作
        price = sym_state.latest_price
        if price <= 0:
            logger.debug(f"[Bot] {cfg.symbol} 價格未更新 ({price})，跳過本次網格操作")
            return
        precision = self.precision_info.get(ccxt_sym, {"price": 4, "amount": 0, "min_notional": 5})
        min_notional = precision.get("min_notional", 5) * 1.1  # +10% 緩衝，避免邊界問題
        
        # 動態計算滿足 min_notional 的最小數量
        import math
        min_qty_for_notional = min_notional / price if price > 0 else 1
        # 向上取整到精度
        amount_prec = precision.get("amount", 0)
        if amount_prec > 0:
            factor = 10 ** amount_prec
            base_qty = math.ceil(max(cfg.initial_quantity, min_qty_for_notional) * factor) / factor
        else:
            base_qty = math.ceil(max(cfg.initial_quantity, min_qty_for_notional))
        
        long_pos = sym_state.long_position
        short_pos = sym_state.short_position
        
        # 計算實際名義價值
        notional = base_qty * price
        
        tp_spacing, gs_spacing = self._get_dynamic_spacing(cfg, sym_state)

        # 計算策略決策
        long_decision = GridStrategy.get_grid_decision(
            price=price, my_position=long_pos, opposite_position=short_pos,
            position_threshold=cfg.position_threshold, position_limit=cfg.position_limit,
            base_qty=base_qty, take_profit_spacing=tp_spacing, grid_spacing=gs_spacing, side='long')
        short_decision = GridStrategy.get_grid_decision(
            price=price, my_position=short_pos, opposite_position=long_pos,
            position_threshold=cfg.position_threshold, position_limit=cfg.position_limit,
            base_qty=base_qty, take_profit_spacing=tp_spacing, grid_spacing=gs_spacing, side='short')
        sym_state.long_dead_mode = long_decision['dead_mode']
        sym_state.short_dead_mode = short_decision['dead_mode']

        try:
            # === 多頭處理 ===
            await self.cancel_orders_for_side(ccxt_sym, 'long')
            
            if long_pos > 0:
                # 有持倉: 下止盈單
                tp_price = round(long_decision['tp_price'], precision['price'])
                tp_qty = round(min(long_decision['tp_qty'], long_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', tp_qty, tp_price,
                        position_side='LONG', reduce_only=True
                    )
            
            # 補倉單 (或無倉時的開倉單)
            if not long_decision['dead_mode'] and long_decision['entry_price']:
                entry_price = round(long_decision['entry_price'], precision['price'])
                entry_qty = round(long_decision['entry_qty'], precision['amount'])
                # 無倉時使用 best_bid，有倉時使用計算的 entry_price
                if long_pos == 0 and sym_state.best_bid > 0:
                    entry_price = round(sym_state.best_bid, precision['price'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', entry_qty, entry_price,
                        position_side='LONG', reduce_only=False
                    )

            # === 空頭處理 ===
            await self.cancel_orders_for_side(ccxt_sym, 'short')
            
            if short_pos > 0:
                # 有持倉: 下止盈單
                tp_price = round(short_decision['tp_price'], precision['price'])
                tp_qty = round(min(short_decision['tp_qty'], short_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', tp_qty, tp_price,
                        position_side='SHORT', reduce_only=True
                    )
            
            # 補倉單 (或無倉時的開倉單)
            if not short_decision['dead_mode'] and short_decision['entry_price']:
                entry_price = round(short_decision['entry_price'], precision['price'])
                entry_qty = round(short_decision['entry_qty'], precision['amount'])
                # 無倉時使用 best_ask，有倉時使用計算的 entry_price
                if short_pos == 0 and sym_state.best_ask > 0:
                    entry_price = round(sym_state.best_ask, precision['price'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', entry_qty, entry_price,
                        position_side='SHORT', reduce_only=False
                    )
                    
        except Exception as e:
            logger.error(f"[Bot] {cfg.symbol} 下單失敗: {e}")

    async def cancel_orders_for_side(self, symbol: str, position_side: str):
        """
        取消指定方向的掛單 (與 Terminal 版一致)
        
        精確判斷取消邏輯:
        - position_side='long': 取消 BUY LONG (補倉) 和 SELL LONG (止盈) 訂單
        - position_side='short': 取消 SELL SHORT (補倉) 和 BUY SHORT (止盈) 訂單
        
        Args:
            symbol: CCXT 交易對 (如 XRP/USDC:USDC)
            position_side: 'long' 或 'short'
        """
        try:
            orders = self.adapter.fetch_open_orders(symbol)
            for order in orders:
                order_side = order.get('side', '').lower()
                order_pos_side = order.get('info', {}).get('positionSide', 'BOTH').upper()
                reduce_only = order.get('reduceOnly', False) or order.get('info', {}).get('reduceOnly', False)

                should_cancel = False
                if position_side.lower() == 'long':
                    # 多頭補倉單: buy + LONG + 非減倉
                    # 多頭止盈單: sell + LONG + 減倉
                    if (not reduce_only and order_side == 'buy' and order_pos_side == 'LONG') or \
                       (reduce_only and order_side == 'sell' and order_pos_side == 'LONG'):
                        should_cancel = True
                elif position_side.lower() == 'short':
                    # 空頭補倉單: sell + SHORT + 非減倉
                    # 空頭止盈單: buy + SHORT + 減倉
                    if (not reduce_only and order_side == 'sell' and order_pos_side == 'SHORT') or \
                       (reduce_only and order_side == 'buy' and order_pos_side == 'SHORT'):
                        should_cancel = True

                if should_cancel:
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], symbol)
        except Exception as e:
            logger.error(f"撤單失敗 {symbol}: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # MAX 增強功能 (與 Terminal 版一致)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_dynamic_spacing(self, cfg, sym_state) -> tuple:
        """
        獲取動態調整後的間距 (與 Terminal 版一致)
        
        優先順序:
        1. 領先指標 (OFI, Volume, Spread) - 預測未來波動
        2. 動態網格 (ATR) - 滯後指標作為備用
        3. GLFT 偏移 - 庫存控制
        """
        max_cfg = self.config.max_enhancement
        ccxt_symbol = cfg.ccxt_symbol
        
        # 基礎間距
        base_take_profit = cfg.take_profit_spacing
        base_grid_spacing = cfg.grid_spacing
        
        # === 1. Bandit 參數 ===
        if self.config.bandit.enabled:
            params = self.bandit_optimizer.get_current_params()
            base_take_profit = params.take_profit_spacing
            base_grid_spacing = params.grid_spacing
        
        # === 2. 領先指標調整 (優先於 ATR) ===
        leading_reason = ""
        if self.config.leading_indicator.enabled:
            # 檢查是否應該暫停交易 (極端情況)
            should_pause, pause_reason = self.leading_indicator.should_pause_trading(ccxt_symbol)
            if should_pause:
                logger.warning(f"[LeadingIndicator] {cfg.symbol} 暫停交易: {pause_reason}")
                base_take_profit *= 2.0
                base_grid_spacing *= 2.0
                leading_reason = f"暫停:{pause_reason}"
            else:
                # 正常領先指標調整
                adjusted_spacing, leading_reason = self.leading_indicator.get_spacing_adjustment(
                    ccxt_symbol, base_grid_spacing
                )
                if adjusted_spacing != base_grid_spacing:
                    ratio = adjusted_spacing / base_grid_spacing
                    base_grid_spacing = adjusted_spacing
                    base_take_profit *= ratio  # 等比例調整止盈
        
        # === 3. 動態網格範圍 (ATR - 滯後指標) ===
        if not leading_reason or leading_reason == "正常":
            take_profit, grid_spacing = self.dynamic_grid.get_dynamic_spacing(
                ccxt_symbol,
                base_take_profit,
                base_grid_spacing,
                max_cfg
            )
        else:
            take_profit = base_take_profit
            grid_spacing = base_grid_spacing
        
        # === 4. GLFT 偏移 (根據庫存調整) ===
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
        
        return take_profit, grid_spacing

    def _get_adjusted_quantity(self, cfg, sym_state, side: str, is_take_profit: bool) -> float:
        """
        獲取調整後的數量 (與 Terminal 版一致)
        整合: Funding Rate 偏向 + GLFT 數量調整 + 原有邏輯
        """
        max_cfg = self.config.max_enhancement
        base_qty = cfg.initial_quantity
        
        # 1. 原有邏輯: position_limit / position_threshold
        if is_take_profit:
            if side == 'long':
                if sym_state.long_position > cfg.position_limit:
                    base_qty *= 2
                elif sym_state.short_position >= cfg.position_threshold:
                    base_qty *= 2
            else:
                if sym_state.short_position > cfg.position_limit:
                    base_qty *= 2
                elif sym_state.long_position >= cfg.position_threshold:
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
                cfg.ccxt_symbol, max_cfg
            )
            if side == 'long':
                base_qty *= long_bias
            else:
                base_qty *= short_bias
        
        return max(cfg.initial_quantity * 0.5, base_qty)

    async def _check_and_reduce_positions(self):
        """
        檢查並減倉 (雙向持倉過大時) - 與 Terminal 版一致
        """
        REDUCE_COOLDOWN = 60  # 減倉冷卻時間
        
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            
            ccxt_symbol = sym_config.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_symbol)
            if not sym_state:
                continue
            
            local_threshold = sym_config.position_threshold * 0.8
            reduce_qty = sym_config.position_threshold * 0.1
            
            # 冷卻時間檢查
            last_reduce = self.state.last_reduce_time.get(ccxt_symbol, 0)
            import time
            if time.time() - last_reduce < REDUCE_COOLDOWN:
                continue
            
            # 雙向持倉都過大時減倉
            if (sym_state.long_position >= local_threshold and 
                sym_state.short_position >= local_threshold):
                logger.info(f"[風控] {sym_config.symbol} 多空持倉均超過 {local_threshold}，開始雙向減倉")
                
                try:
                    # 市價平多
                    if sym_state.long_position > 0:
                        await asyncio.to_thread(
                            self.adapter.create_market_order,
                            ccxt_symbol, 'sell', reduce_qty,
                            position_side='LONG', reduce_only=True
                        )
                        logger.info(f"[風控] {sym_config.symbol} 市價平多 {reduce_qty}")
                    
                    # 市價平空
                    if sym_state.short_position > 0:
                        await asyncio.to_thread(
                            self.adapter.create_market_order,
                            ccxt_symbol, 'buy', reduce_qty,
                            position_side='SHORT', reduce_only=True
                        )
                        logger.info(f"[風控] {sym_config.symbol} 市價平空 {reduce_qty}")
                    
                    self.state.last_reduce_time[ccxt_symbol] = time.time()
                except Exception as e:
                    logger.error(f"[風控] 減倉失敗: {e}")

    def _should_adjust_grid(self, cfg, sym_state, side: str) -> bool:
        """
        檢查是否需要調整網格 (與 Terminal 版一致)
        避免頻繁取消重掛訂單
        """
        price = sym_state.latest_price
        deviation_threshold = cfg.grid_spacing * 0.5
        
        if side == 'long':
            # 沒有掛單時必須調整
            if sym_state.buy_long_orders <= 0 or sym_state.sell_long_orders <= 0:
                return True
            # 價格偏離過大時調整
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

    # ═══════════════════════════════════════════════════════════════════════════
    # 風控邏輯
    # ═══════════════════════════════════════════════════════════════════════════

    async def _risk_monitor_loop(self):
        """風控監控循環 - 在 _sync_loop 中呼叫"""
        if not self.config.risk.enabled:
            return

        # 檢查保證金使用率
        await self._check_margin_threshold()

        # 檢查硬止損 (防爆倉)
        await self._check_hard_stop()

        # 檢查追蹤止盈
        await self._check_trailing_stop()

    async def _check_margin_threshold(self):
        """檢查保證金閾值"""
        risk = self.config.risk
        for currency in ["USDC", "USDT"]:
            acc = self.state.get_account(currency)
            if acc.equity <= 0:
                continue

            margin_ratio = acc.margin_ratio
            if margin_ratio >= risk.margin_threshold:
                logger.warning(f"[Risk] {currency} 保證金使用率 {margin_ratio*100:.1f}% 超過閾值 {risk.margin_threshold*100:.0f}%")
                # 可以在這裡觸發減倉邏輯

    async def _check_hard_stop(self):
        """
        硬止損檢查 - 防止爆倉

        當單方向浮虧超過帳戶權益的 max_loss_pct 時，強制平倉該方向
        例如: max_loss_pct = 0.03 (3%)，帳戶權益 $1000
              多單浮虧 > $30 時，強制平倉所有多單
        """
        risk = self.config.risk
        if not risk.hard_stop_enabled:
            return

        # 計算帳戶總權益
        total_equity = 0
        for currency in ["USDC", "USDT"]:
            acc = self.state.get_account(currency)
            total_equity += acc.equity

        if total_equity <= 0:
            return

        max_loss = total_equity * risk.max_loss_pct

        # 檢查每個交易對的多空方向浮虧
        for cfg in self.config.symbols.values():
            if not cfg.enabled:
                continue

            ccxt_sym = cfg.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_sym)
            if not sym_state:
                continue

            price = sym_state.latest_price
            if price <= 0:
                continue

            # 計算多單浮虧 (負數表示虧損)
            if sym_state.long_position > 0 and sym_state.long_avg_price > 0:
                long_pnl = (price - sym_state.long_avg_price) * sym_state.long_position
                if long_pnl < -max_loss:
                    logger.warning(f"[HardStop] {cfg.symbol} 多單浮虧 ${abs(long_pnl):.2f} 超過閾值 ${max_loss:.2f}")
                    await self._emergency_close_side(cfg, "long")

            # 計算空單浮虧 (負數表示虧損)
            if sym_state.short_position > 0 and sym_state.short_avg_price > 0:
                short_pnl = (sym_state.short_avg_price - price) * sym_state.short_position
                if short_pnl < -max_loss:
                    logger.warning(f"[HardStop] {cfg.symbol} 空單浮虧 ${abs(short_pnl):.2f} 超過閾值 ${max_loss:.2f}")
                    await self._emergency_close_side(cfg, "short")

    async def _emergency_close_side(self, cfg, side: str):
        """
        緊急平倉單一方向

        Args:
            cfg: SymbolConfig
            side: "long" 或 "short"
        """
        ccxt_sym = cfg.ccxt_symbol
        sym_state = self.state.symbols.get(ccxt_sym)
        if not sym_state:
            return

        try:
            # 取消該方向的掛單
            orders = self.adapter.fetch_open_orders(ccxt_sym)
            for order in orders:
                order_side = order.get('side', '').lower()
                # 多單平倉掛的是 sell 單，空單平倉掛的是 buy 單
                if side == "long" and order_side == "sell":
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)
                elif side == "short" and order_side == "buy":
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)

            # 市價平倉
            if side == "long" and sym_state.long_position > 0:
                await asyncio.to_thread(
                    self.adapter.create_market_order,
                    ccxt_sym, 'sell',
                    sym_state.long_position,
                    position_side=side.upper(),
                    reduce_only=True
                )
                logger.warning(f"[HardStop] {cfg.symbol} 強制平多倉 {sym_state.long_position}")
            
            elif side == "short" and sym_state.short_position > 0:
                await asyncio.to_thread(
                    self.adapter.create_market_order,
                    ccxt_sym, 'buy',
                    sym_state.short_position,
                    position_side=side.upper(),
                    reduce_only=True
                )
                logger.warning(f"[HardStop] {cfg.symbol} 強制平空倉 {sym_state.short_position}")

        except Exception as e:
            logger.error(f"[HardStop] {cfg.symbol} {side} 平倉失敗: {e}")

    async def _check_trailing_stop(self):
        """檢查追蹤止盈"""
        risk = self.config.risk
        total_pnl = self.state.total_unrealized_pnl

        # 更新峰值
        if total_pnl > self.state.peak_equity:
            self.state.peak_equity = total_pnl

        # 檢查是否達到啟動條件
        if total_pnl >= risk.trailing_start_profit:
            # 計算回撤
            drawdown = self.state.peak_equity - total_pnl

            # 動態回撤閾值 (最大 10% 或最小 $2)
            drawdown_threshold = max(
                self.state.peak_equity * risk.trailing_drawdown_pct,
                risk.trailing_min_drawdown
            )

            if drawdown >= drawdown_threshold:
                logger.warning(f"[Risk] 追蹤止盈觸發! 峰值={self.state.peak_equity:.2f}, "
                             f"當前={total_pnl:.2f}, 回撤={drawdown:.2f}")
                # 觸發市價平倉
                await self._emergency_close_all()

    async def _emergency_close_all(self):
        """緊急平倉所有持倉"""
        logger.warning("[Risk] 執行緊急平倉!")

        for cfg in self.config.symbols.values():
            if not cfg.enabled:
                continue

            ccxt_sym = cfg.ccxt_symbol
            sym_state = self.state.symbols.get(ccxt_sym)
            if not sym_state:
                continue

            try:
                # 取消所有掛單
                orders = self.adapter.fetch_open_orders(ccxt_sym)
                for order in orders:
                    await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)

                # 市價平多倉
                if sym_state.long_position > 0:
                    await asyncio.to_thread(
                        self.adapter.create_market_order,
                        ccxt_sym, 'sell',
                        sym_state.long_position,
                        position_side='LONG',
                        reduce_only=True
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平多倉 {sym_state.long_position}")

                # 市價平空倉
                if sym_state.short_position > 0:
                    await asyncio.to_thread(
                        self.adapter.create_market_order,
                        ccxt_sym, 'buy',
                        sym_state.short_position,
                        position_side='SHORT',
                        reduce_only=True
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平空倉 {sym_state.short_position}")

            except Exception as e:
                logger.error(f"[Risk] {cfg.symbol} 緊急平倉失敗: {e}")

        # 停止交易
        self._stop_event.set()
        self.state.running = False
        logger.warning("[Risk] 交易已停止")
