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
from exchanges import get_adapter, ExchangeAdapter
from exchanges.base import WSMessageType

logger = logging.getLogger("as_grid_max")


def normalize_symbol(symbol: str) -> str:
    """
    標準化交易對符號，處理各交易所不同格式

    Examples:
        XRPUSDT -> XRPUSDT
        XRP_USDT -> XRPUSDT
        XRP/USDT:USDT -> XRPUSDT
        xrpusdt -> XRPUSDT
    """
    # 移除分隔符並轉大寫
    s = symbol.upper().replace("_", "").replace("/", "").replace(":", "")

    # 處理 CCXT 格式中重複的結算幣種 (如 XRPUSDTUSDT -> XRPUSDT)
    for quote in ["USDT", "USDC", "BUSD"]:
        if s.endswith(quote + quote):
            s = s[:-len(quote)]
            break

    return s


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
        self.adapter.init_exchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=testnet
        )
        self.adapter.load_markets()

        # === 舊版兼容: 保留 self.exchange 引用 ===
        self.exchange = self.adapter.exchange

        # 載入精度資訊
        for sym_cfg in self.config.symbols.values():
            if not sym_cfg.enabled:
                continue
            try:
                precision = self.adapter.get_precision(sym_cfg.ccxt_symbol)
                self.precision_info[sym_cfg.ccxt_symbol] = {
                    "price": precision.price_precision,
                    "amount": precision.amount_precision,
                    "min_notional": precision.min_notional
                }
            except Exception as e:
                logger.error(f"獲取 {sym_cfg.ccxt_symbol} 精度失敗: {e}")

        self.funding_manager = FundingRateManager(self.exchange)
        logger.info(f"[Bot] {self.adapter.get_display_name()} 初始化完成，{len(self.precision_info)} 個交易對")

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
                normalized_sym = normalize_symbol(pos.symbol)
                ccxt_symbol = None
                for cfg in self.config.symbols.values():
                    cfg_normalized = normalize_symbol(cfg.symbol)
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

    async def run(self):
        self._init_exchange()
        self._init_state()
        await self._sync_positions()
        for cfg in self.config.symbols.values():
            if cfg.enabled:
                try:
                    self.adapter.set_leverage(cfg.ccxt_symbol, cfg.leverage)
                except Exception as e:
                    logger.warning(f"設置 {cfg.symbol} 槓桿失敗: {e}")
        # 掛初始網格
        await self._place_initial_grids()
        logger.info("[Bot] 啟動 WebSocket...")
        await self._websocket_loop()

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

    async def _websocket_loop(self):
        """WebSocket 主循環 (使用 Adapter 構建 URL)"""
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

                async with websockets.connect(url, ssl=ssl_context, ping_interval=30, ping_timeout=10) as ws:
                    self.ws = ws
                    self.state.connected = True
                    logger.info(f"[Bot] WebSocket 已連接 ({self.adapter.get_display_name()})")
                    asyncio.create_task(self._sync_loop())
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("[Bot] WebSocket 連接關閉，重連中...")
            except Exception as e:
                logger.error(f"[Bot] WebSocket 錯誤: {e}")
            self.state.connected = False
            await asyncio.sleep(5)

    async def _sync_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.sync_interval)
            await self._sync_positions()
            # 風控檢查
            await self._risk_monitor_loop()

    async def _handle_message(self, message: str):
        """處理 WebSocket 消息 (使用 Adapter 解析)"""
        try:
            ws_msg = self.adapter.parse_ws_message(message)
            if not ws_msg:
                return

            if ws_msg.msg_type == WSMessageType.TICKER:
                await self._handle_ticker_update(ws_msg.symbol, ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ORDER_UPDATE:
                await self._handle_order_update(ws_msg.data)
            elif ws_msg.msg_type == WSMessageType.ACCOUNT_UPDATE:
                await self._handle_account_update(ws_msg.data)

        except Exception as e:
            logger.error(f"處理訊息錯誤: {e}")

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
        normalized_symbol = normalize_symbol(symbol)
        for cfg in self.config.symbols.values():
            cfg_normalized = normalize_symbol(cfg.symbol)
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
                cfg_normalized = normalize_symbol(cfg.symbol)
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
        # 標準化 symbol 進行比對
        normalized_raw = normalize_symbol(raw_symbol)
        for cfg in self.config.symbols.values():
            cfg_normalized = normalize_symbol(cfg.symbol)
            if cfg_normalized == normalized_raw and cfg.ccxt_symbol in self.state.symbols:
                sym_state = self.state.symbols[cfg.ccxt_symbol]
                sym_state.latest_price = ticker_update.price
                sym_state.best_bid = ticker_update.bid
                sym_state.best_ask = ticker_update.ask

                # 更新領先指標
                self.leading_indicator.update_spread(cfg.ccxt_symbol, sym_state.best_bid, sym_state.best_ask)
                self.dynamic_grid.update_price(cfg.ccxt_symbol, sym_state.latest_price)

                if self.config.bandit.contextual_enabled:
                    self.bandit_optimizer.update_price(sym_state.latest_price)

                # 間隔控制
                now = time.time()
                last = self.last_grid_time.get(cfg.ccxt_symbol, 0)
                if now - last >= self.grid_interval:
                    self.last_grid_time[cfg.ccxt_symbol] = now
                    await self._place_grid(cfg)
                break

    async def _place_grid(self, cfg):
        ccxt_sym = cfg.ccxt_symbol
        sym_state = self.state.symbols.get(ccxt_sym)
        if not sym_state or sym_state.latest_price <= 0:
            return
        precision = self.precision_info.get(ccxt_sym, {"price": 4, "amount": 0})
        price = sym_state.latest_price
        base_qty = cfg.initial_quantity
        long_pos = sym_state.long_position
        short_pos = sym_state.short_position
        tp_spacing, gs_spacing = cfg.take_profit_spacing, cfg.grid_spacing
        if self.config.bandit.enabled:
            params = self.bandit_optimizer.get_current_params()
            tp_spacing = params.take_profit_spacing
            gs_spacing = params.grid_spacing
        if self.config.leading_indicator.enabled:
            adjusted, reason = self.leading_indicator.get_spacing_adjustment(ccxt_sym, gs_spacing)
            gs_spacing = adjusted
        try:
            orders = self.adapter.fetch_open_orders(ccxt_sym)
            for order in orders:
                await asyncio.to_thread(self.adapter.cancel_order, order['id'], ccxt_sym)
        except Exception:
            pass
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
            # 使用 Adapter 統一下單介面
            if long_pos > 0:
                tp_price = round(long_decision['tp_price'], precision['price'])
                tp_qty = round(min(long_decision['tp_qty'], long_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', tp_qty, tp_price,
                        position_side='LONG', reduce_only=True
                    )
            if not long_decision['dead_mode'] and long_decision['entry_price']:
                entry_price = round(long_decision['entry_price'], precision['price'])
                entry_qty = round(long_decision['entry_qty'], precision['amount'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', entry_qty, entry_price,
                        position_side='LONG', reduce_only=False
                    )
            if short_pos > 0:
                tp_price = round(short_decision['tp_price'], precision['price'])
                tp_qty = round(min(short_decision['tp_qty'], short_pos), precision['amount'])
                if tp_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'buy', tp_qty, tp_price,
                        position_side='SHORT', reduce_only=True
                    )
            if not short_decision['dead_mode'] and short_decision['entry_price']:
                entry_price = round(short_decision['entry_price'], precision['price'])
                entry_qty = round(short_decision['entry_qty'], precision['amount'])
                if entry_qty > 0:
                    await asyncio.to_thread(
                        self.adapter.create_limit_order,
                        ccxt_sym, 'sell', entry_qty, entry_price,
                        position_side='SHORT', reduce_only=False
                    )
        except Exception as e:
            logger.error(f"[Bot] {cfg.symbol} 下單失敗: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 風控邏輯
    # ═══════════════════════════════════════════════════════════════════════════

    async def _risk_monitor_loop(self):
        """風控監控循環 - 在 _sync_loop 中呼叫"""
        if not self.config.risk.enabled:
            return

        # 檢查保證金使用率
        await self._check_margin_threshold()

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
                orders = self.exchange.fetch_open_orders(ccxt_sym)
                for order in orders:
                    await asyncio.to_thread(self.exchange.cancel_order, order['id'], ccxt_sym)

                # 市價平多倉
                if sym_state.long_position > 0:
                    await asyncio.to_thread(
                        self.exchange.create_order,
                        ccxt_sym, 'market', 'sell',
                        sym_state.long_position,
                        None,
                        {'reduceOnly': True}
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平多倉 {sym_state.long_position}")

                # 市價平空倉
                if sym_state.short_position > 0:
                    await asyncio.to_thread(
                        self.exchange.create_order,
                        ccxt_sym, 'market', 'buy',
                        sym_state.short_position,
                        None,
                        {'reduceOnly': True}
                    )
                    logger.info(f"[Risk] {cfg.symbol} 平空倉 {sym_state.short_position}")

            except Exception as e:
                logger.error(f"[Risk] {cfg.symbol} 緊急平倉失敗: {e}")

        # 停止交易
        self._stop_event.set()
        self.state.running = False
        logger.warning("[Risk] 交易已停止")
