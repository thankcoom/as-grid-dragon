"""
回測引擎核心模組
網格交易策略回測器

整合 GridStrategy 邏輯，確保回測與實盤一致：
- 裝死模式 (Dead Mode)
- 持倉閾值控制
- 止盈加倍機制
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .config import Config

# 從 core.strategy 導入統一的 GridStrategy (避免重複代碼)
# 路徑設置已在 __init__.py 中處理
from core.strategy import GridStrategy


@dataclass
class Position:
    """持倉資訊"""
    entry_price: float
    quantity: float
    margin: float
    side: str  # "long" or "short"
    entry_time: datetime = None


@dataclass
class Trade:
    """交易記錄"""
    timestamp: datetime
    action: str  # BUY, SELL, SELL_SHORT, COVER_SHORT
    price: float
    quantity: float
    side: str  # LONG, SHORT
    pnl: float
    fee: float
    gross_pnl: float
    unrealized_pnl: float
    equity: float


@dataclass
class BacktestResult:
    """回測結果"""
    final_equity: float
    return_pct: float
    max_drawdown: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    trades_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    direction: str
    config: Config
    trade_history: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple] = field(default_factory=list)

    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            "final_equity": self.final_equity,
            "return_pct": self.return_pct,
            "max_drawdown": self.max_drawdown,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "trades_count": self.trades_count,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "direction": self.direction
        }

    def __str__(self) -> str:
        return (
            f"回測結果 ({self.config.symbol})\n"
            f"{'='*40}\n"
            f"最終淨值: ${self.final_equity:.2f}\n"
            f"收益率: {self.return_pct*100:.2f}%\n"
            f"最大回撤: {self.max_drawdown*100:.2f}%\n"
            f"交易次數: {self.trades_count}\n"
            f"勝率: {self.win_rate*100:.1f}%\n"
            f"盈虧比: {self.profit_factor:.2f}\n"
            f"已實現盈虧: ${self.realized_pnl:.2f}\n"
            f"未實現盈虧: ${self.unrealized_pnl:.2f}"
        )


class GridBacktester:
    """
    網格交易回測器

    整合 GridStrategy 邏輯，確保與實盤一致：
    - 裝死模式判斷
    - 持倉閾值控制
    - 止盈加倍機制
    """

    def __init__(self, df: pd.DataFrame, config: Config):
        """
        初始化回測器

        Args:
            df: K線數據 DataFrame (需含 open_time, open, high, low, close, volume)
            config: 回測配置
        """
        self.df = df.reset_index(drop=True)
        self.config = config

        # 網格設定
        self.long_settings = config.long_settings
        self.short_settings = config.short_settings

        # 帳戶狀態
        self.balance = config.initial_balance
        self.max_equity = config.initial_balance

        # 持倉
        self.long_positions: List[Position] = []
        self.short_positions: List[Position] = []

        # 掛單
        self.orders = {"long": [], "short": []}

        # 記錄
        self.trade_history: List[Trade] = []
        self.equity_curve: List[Tuple] = []

        # 時間追蹤
        self.last_refresh_time = None
        self.last_long_price = None
        self.last_short_price = None

        # 裝死模式追蹤
        self.long_dead_mode = False
        self.short_dead_mode = False

        # 初始化網格
        initial_price = self.df['close'].iloc[0]
        self._init_orders(initial_price)

    def _init_orders(self, price: float):
        """初始化網格訂單"""
        if self.config.direction in ["long", "both"]:
            self._place_long_orders(price)
        if self.config.direction in ["short", "both"]:
            self._place_short_orders(price)

    def _place_long_orders(self, current_price: float):
        """
        多頭網格：上方止盈(小間距)，下方補倉(大間距)
        """
        self.orders["long"] = [
            (current_price * (1 - self.long_settings["down_spacing"]), "BUY"),
            (current_price * (1 + self.long_settings["up_spacing"]), "SELL")
        ]
        self.last_long_price = current_price

    def _place_short_orders(self, current_price: float):
        """
        空頭網格：上方補倉(大間距)，下方止盈(小間距)
        """
        self.orders["short"] = [
            (current_price * (1 + self.short_settings["up_spacing"]), "SELL_SHORT"),
            (current_price * (1 - self.short_settings["down_spacing"]), "COVER_SHORT")
        ]
        self.last_short_price = current_price

    def _refresh_orders_if_needed(self, price: float, current_time: datetime):
        """定期刷新網格"""
        if self.last_refresh_time is None:
            self.last_refresh_time = current_time
            return

        interval = timedelta(minutes=self.config.grid_refresh_interval)
        if (current_time - self.last_refresh_time) >= interval:
            if self.config.direction in ["long", "both"]:
                self._place_long_orders(price)
            if self.config.direction in ["short", "both"]:
                self._place_short_orders(price)
            self.last_refresh_time = current_time

    def _calculate_unrealized_pnl(self, price: float) -> float:
        """計算未實現盈虧"""
        long_pnl = sum(
            (price - pos.entry_price) * pos.quantity
            for pos in self.long_positions
        )
        short_pnl = sum(
            (pos.entry_price - price) * pos.quantity
            for pos in self.short_positions
        )
        return long_pnl + short_pnl

    def _get_available_margin(self) -> float:
        """計算可用保證金"""
        used_margin = sum(pos.margin for pos in self.long_positions + self.short_positions)
        return self.balance - used_margin

    def _process_long_orders(self, price: float, timestamp: datetime, available_margin: float) -> float:
        """
        處理多頭訂單 - 使用 GridStrategy 統一邏輯

        整合裝死模式：
        - 持倉超過 position_threshold 時停止補倉
        - 使用特殊止盈價格
        - 止盈數量可能加倍
        """
        effective_value = self.config.order_value * self.config.leverage
        base_qty = effective_value / price

        # 計算當前持倉量
        long_position = sum(pos.quantity for pos in self.long_positions)
        short_position = sum(pos.quantity for pos in self.short_positions)

        # 使用 GridStrategy 獲取決策
        decision = GridStrategy.get_grid_decision(
            price=self.last_long_price or price,
            my_position=long_position,
            opposite_position=short_position,
            position_threshold=self.config.position_threshold,
            position_limit=self.config.position_limit,
            base_qty=base_qty,
            take_profit_spacing=self.config.take_profit_spacing,
            grid_spacing=self.config.grid_spacing,
            side='long',
            dead_mode_enabled=getattr(self.config, 'dead_mode_enabled', True),
            fallback_long=getattr(self.config, 'dead_mode_fallback_long', 1.05),
            fallback_short=getattr(self.config, 'dead_mode_fallback_short', 0.95)
        )

        self.long_dead_mode = decision['dead_mode']
        tp_price = decision['tp_price']
        entry_price = decision['entry_price']
        tp_qty = decision['tp_qty']

        # 補倉邏輯 (非裝死模式)
        if not decision['dead_mode'] and entry_price and price <= entry_price:
            qty = base_qty
            margin_required = (qty * price) / self.config.leverage
            fee_cost = qty * price * (self.config.fee_pct / 2)

            if (margin_required + fee_cost) <= available_margin:
                self.balance -= (margin_required + fee_cost)
                self.long_positions.append(Position(
                    entry_price=price,
                    quantity=qty,
                    margin=margin_required,
                    side="long",
                    entry_time=timestamp
                ))

                unrealized = self._calculate_unrealized_pnl(price)
                equity = self.balance + unrealized

                self.trade_history.append(Trade(
                    timestamp=timestamp,
                    action="BUY",
                    price=price,
                    quantity=qty,
                    side="LONG",
                    pnl=0.0,
                    fee=fee_cost,
                    gross_pnl=0.0,
                    unrealized_pnl=unrealized,
                    equity=equity
                ))

                self.last_long_price = price
                return available_margin - margin_required - fee_cost

        # 止盈邏輯 (兩種模式都執行)
        if self.long_positions and price >= tp_price:
            # 根據止盈數量決定平倉多少
            remaining_tp = tp_qty
            total_pnl = 0

            while self.long_positions and remaining_tp > 0:
                pos = self.long_positions[0]
                if pos.quantity <= remaining_tp:
                    # 全部平倉
                    self.long_positions.pop(0)
                    fee_cost = pos.quantity * price * (self.config.fee_pct / 2)
                    gross_pnl = (price - pos.entry_price) * pos.quantity
                    net_pnl = gross_pnl - fee_cost
                    self.balance += pos.margin + net_pnl
                    total_pnl += net_pnl

                    self.trade_history.append(Trade(
                        timestamp=timestamp,
                        action="SELL",
                        price=price,
                        quantity=pos.quantity,
                        side="LONG",
                        pnl=net_pnl,
                        fee=fee_cost,
                        gross_pnl=gross_pnl,
                        unrealized_pnl=self._calculate_unrealized_pnl(price),
                        equity=self.balance + self._calculate_unrealized_pnl(price)
                    ))

                    remaining_tp -= pos.quantity
                    available_margin += pos.margin + net_pnl
                else:
                    # 部分平倉
                    close_ratio = remaining_tp / pos.quantity
                    close_qty = remaining_tp
                    close_margin = pos.margin * close_ratio
                    fee_cost = close_qty * price * (self.config.fee_pct / 2)
                    gross_pnl = (price - pos.entry_price) * close_qty
                    net_pnl = gross_pnl - fee_cost
                    self.balance += close_margin + net_pnl
                    total_pnl += net_pnl

                    self.trade_history.append(Trade(
                        timestamp=timestamp,
                        action="SELL",
                        price=price,
                        quantity=close_qty,
                        side="LONG",
                        pnl=net_pnl,
                        fee=fee_cost,
                        gross_pnl=gross_pnl,
                        unrealized_pnl=self._calculate_unrealized_pnl(price),
                        equity=self.balance + self._calculate_unrealized_pnl(price)
                    ))

                    pos.quantity -= close_qty
                    pos.margin -= close_margin
                    available_margin += close_margin + net_pnl
                    remaining_tp = 0

            self.last_long_price = price

        return available_margin

    def _process_short_orders(self, price: float, timestamp: datetime, available_margin: float) -> float:
        """
        處理空頭訂單 - 使用 GridStrategy 統一邏輯

        整合裝死模式：
        - 持倉超過 position_threshold 時停止補倉
        - 使用特殊止盈價格
        - 止盈數量可能加倍
        """
        effective_value = self.config.order_value * self.config.leverage
        base_qty = effective_value / price

        # 計算當前持倉量
        long_position = sum(pos.quantity for pos in self.long_positions)
        short_position = sum(pos.quantity for pos in self.short_positions)

        # 使用 GridStrategy 獲取決策
        decision = GridStrategy.get_grid_decision(
            price=self.last_short_price or price,
            my_position=short_position,
            opposite_position=long_position,
            position_threshold=self.config.position_threshold,
            position_limit=self.config.position_limit,
            base_qty=base_qty,
            take_profit_spacing=self.config.take_profit_spacing,
            grid_spacing=self.config.grid_spacing,
            side='short',
            dead_mode_enabled=getattr(self.config, 'dead_mode_enabled', True),
            fallback_long=getattr(self.config, 'dead_mode_fallback_long', 1.05),
            fallback_short=getattr(self.config, 'dead_mode_fallback_short', 0.95)
        )

        self.short_dead_mode = decision['dead_mode']
        tp_price = decision['tp_price']
        entry_price = decision['entry_price']
        tp_qty = decision['tp_qty']

        # 補倉邏輯 (非裝死模式)
        if not decision['dead_mode'] and entry_price and price >= entry_price:
            qty = base_qty
            margin_required = (qty * price) / self.config.leverage
            fee_cost = qty * price * (self.config.fee_pct / 2)

            if (margin_required + fee_cost) <= available_margin:
                self.balance -= (margin_required + fee_cost)
                self.short_positions.append(Position(
                    entry_price=price,
                    quantity=qty,
                    margin=margin_required,
                    side="short",
                    entry_time=timestamp
                ))

                unrealized = self._calculate_unrealized_pnl(price)
                equity = self.balance + unrealized

                self.trade_history.append(Trade(
                    timestamp=timestamp,
                    action="SELL_SHORT",
                    price=price,
                    quantity=qty,
                    side="SHORT",
                    pnl=0.0,
                    fee=fee_cost,
                    gross_pnl=0.0,
                    unrealized_pnl=unrealized,
                    equity=equity
                ))

                self.last_short_price = price
                return available_margin - margin_required - fee_cost

        # 止盈邏輯 (兩種模式都執行)
        if self.short_positions and price <= tp_price:
            # 根據止盈數量決定平倉多少
            remaining_tp = tp_qty
            total_pnl = 0

            while self.short_positions and remaining_tp > 0:
                pos = self.short_positions[0]
                if pos.quantity <= remaining_tp:
                    # 全部平倉
                    self.short_positions.pop(0)
                    fee_cost = pos.quantity * price * (self.config.fee_pct / 2)
                    gross_pnl = (pos.entry_price - price) * pos.quantity
                    net_pnl = gross_pnl - fee_cost
                    self.balance += pos.margin + net_pnl
                    total_pnl += net_pnl

                    self.trade_history.append(Trade(
                        timestamp=timestamp,
                        action="COVER_SHORT",
                        price=price,
                        quantity=pos.quantity,
                        side="SHORT",
                        pnl=net_pnl,
                        fee=fee_cost,
                        gross_pnl=gross_pnl,
                        unrealized_pnl=self._calculate_unrealized_pnl(price),
                        equity=self.balance + self._calculate_unrealized_pnl(price)
                    ))

                    remaining_tp -= pos.quantity
                    available_margin += pos.margin + net_pnl
                else:
                    # 部分平倉
                    close_ratio = remaining_tp / pos.quantity
                    close_qty = remaining_tp
                    close_margin = pos.margin * close_ratio
                    fee_cost = close_qty * price * (self.config.fee_pct / 2)
                    gross_pnl = (pos.entry_price - price) * close_qty
                    net_pnl = gross_pnl - fee_cost
                    self.balance += close_margin + net_pnl
                    total_pnl += net_pnl

                    self.trade_history.append(Trade(
                        timestamp=timestamp,
                        action="COVER_SHORT",
                        price=price,
                        quantity=close_qty,
                        side="SHORT",
                        pnl=net_pnl,
                        fee=fee_cost,
                        gross_pnl=gross_pnl,
                        unrealized_pnl=self._calculate_unrealized_pnl(price),
                        equity=self.balance + self._calculate_unrealized_pnl(price)
                    ))

                    pos.quantity -= close_qty
                    pos.margin -= close_margin
                    available_margin += close_margin + net_pnl
                    remaining_tp = 0

            self.last_short_price = price

        return available_margin

    def run(self) -> BacktestResult:
        """
        執行回測 - 與終端 UI (as_terminal_max.py) 完全一致的邏輯

        Returns:
            BacktestResult: 回測結果
        """
        # 使用終端 UI 兼容模式
        if self.config.terminal_ui_mode and self.config.initial_quantity > 0:
            return self._run_terminal_ui_mode()
        else:
            return self._run_legacy_mode()

    def _run_terminal_ui_mode(self) -> BacktestResult:
        """
        終端 UI 兼容模式 - 與 as_terminal_max.py BacktestManager.run_backtest 完全一致
        """
        balance = self.config.initial_balance
        max_equity = balance

        long_positions = []
        short_positions = []
        trades = []
        equity_curve = []

        # 關鍵：order_value = 幣數量 × 初始價格 (與終端 UI 一致)
        initial_quantity = self.config.initial_quantity
        order_value = initial_quantity * self.df['close'].iloc[0]
        leverage = self.config.leverage
        fee_pct = self.config.fee_pct

        last_long_price = self.df['close'].iloc[0]
        last_short_price = self.df['close'].iloc[0]

        # 計算持倉閾值 (與終端 UI 一致)
        position_threshold = initial_quantity * self.config.threshold_multiplier
        position_limit = initial_quantity * self.config.limit_multiplier

        for _, row in self.df.iterrows():
            price = row['close']
            timestamp = row.get('open_time', None)

            # 計算當前持倉量
            long_position = sum(p["qty"] for p in long_positions)
            short_position = sum(p["qty"] for p in short_positions)

            # === 多頭網格 (使用 GridStrategy 統一邏輯) ===
            if self.config.direction in ["long", "both"]:
                long_decision = GridStrategy.get_grid_decision(
                    price=last_long_price,
                    my_position=long_position,
                    opposite_position=short_position,
                    position_threshold=position_threshold,
                    position_limit=position_limit,
                    base_qty=initial_quantity,
                    take_profit_spacing=self.config.take_profit_spacing,
                    grid_spacing=self.config.grid_spacing,
                    side='long'
                )

                sell_price = long_decision['tp_price']
                long_tp_qty = long_decision['tp_qty']
                long_dead_mode = long_decision['dead_mode']
                buy_price = long_decision['entry_price'] if long_decision['entry_price'] else last_long_price * (1 - self.config.grid_spacing)

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
                            trades.append({"pnl": net_pnl, "type": "long", "timestamp": timestamp})
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
                            trades.append({"pnl": net_pnl, "type": "long", "timestamp": timestamp})
                            pos["qty"] -= close_qty
                            pos["margin"] -= close_margin
                            remaining_tp = 0
                    last_long_price = price

            # === 空頭網格 (使用 GridStrategy 統一邏輯) ===
            if self.config.direction in ["short", "both"]:
                short_decision = GridStrategy.get_grid_decision(
                    price=last_short_price,
                    my_position=short_position,
                    opposite_position=long_position,
                    position_threshold=position_threshold,
                    position_limit=position_limit,
                    base_qty=initial_quantity,
                    take_profit_spacing=self.config.take_profit_spacing,
                    grid_spacing=self.config.grid_spacing,
                    side='short'
                )

                cover_price = short_decision['tp_price']
                short_tp_qty = short_decision['tp_qty']
                short_dead_mode = short_decision['dead_mode']
                sell_short_price = short_decision['entry_price'] if short_decision['entry_price'] else last_short_price * (1 + self.config.grid_spacing)

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
                            trades.append({"pnl": net_pnl, "type": "short", "timestamp": timestamp})
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
                            trades.append({"pnl": net_pnl, "type": "short", "timestamp": timestamp})
                            pos["qty"] -= close_qty
                            pos["margin"] -= close_margin
                            remaining_tp = 0
                    last_short_price = price

            # 計算淨值
            unrealized = sum((price - p["price"]) * p["qty"] for p in long_positions)
            unrealized += sum((p["price"] - price) * p["qty"] for p in short_positions)
            equity = balance + unrealized
            max_equity = max(max_equity, equity)
            equity_curve.append((timestamp, price, equity))

        # 計算結果
        final_price = self.df['close'].iloc[-1]
        unrealized_pnl = sum((final_price - p["price"]) * p["qty"] for p in long_positions)
        unrealized_pnl += sum((p["price"] - final_price) * p["qty"] for p in short_positions)

        realized_pnl = sum(t["pnl"] for t in trades)
        final_equity = balance + unrealized_pnl

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] < 0]

        # 計算 Sharpe Ratio (基於權益曲線收益率，年化)
        # 正確公式: Sharpe = (平均收益率 / 收益率標準差) × √(年化因子)
        sharpe_ratio = 0.0
        if len(equity_curve) > 1:
            import numpy as np
            # 計算逐期收益率
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i - 1][2]  # equity_curve 是 (timestamp, price, equity)
                curr_equity = equity_curve[i][2]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)

            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns, ddof=1)

                if std_return > 0:
                    # 1 分鐘 K 線，一年約 525600 分鐘
                    periods_per_year = 525600
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)

        return BacktestResult(
            final_equity=final_equity,
            return_pct=(final_equity - self.config.initial_balance) / self.config.initial_balance,
            max_drawdown=1 - (min(e[2] for e in equity_curve) / max_equity) if equity_curve else 0,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            trades_count=len(trades),
            win_rate=len(winning) / len(trades) if trades else 0,
            profit_factor=sum(t["pnl"] for t in winning) / abs(sum(t["pnl"] for t in losing)) if losing else float('inf'),
            sharpe_ratio=sharpe_ratio,
            direction=self.config.direction,
            config=self.config,
            trade_history=[],  # 簡化版不記錄詳細交易歷史
            equity_curve=equity_curve
        )

    def _run_legacy_mode(self) -> BacktestResult:
        """
        舊版模式 - 保留原有邏輯以向後兼容
        """
        final_price = self.df['close'].iloc[-1]

        for _, row in self.df.iterrows():
            # 檢查持倉上限
            total_positions = len(self.long_positions) + len(self.short_positions)
            if total_positions >= self.config.max_positions:
                break

            price = row['close']
            timestamp = row['open_time']

            # 定期刷新網格
            self._refresh_orders_if_needed(price, timestamp)

            # 取得可用保證金
            available_margin = self._get_available_margin()

            # 處理多頭訂單
            if self.config.direction in ["long", "both"]:
                available_margin = self._process_long_orders(price, timestamp, available_margin)

            # 處理空頭訂單
            if self.config.direction in ["short", "both"]:
                available_margin = self._process_short_orders(price, timestamp, available_margin)

            # 計算當前淨值
            unrealized_pnl = self._calculate_unrealized_pnl(price)
            equity = self.balance + unrealized_pnl
            self.max_equity = max(self.max_equity, equity)

            # 記錄淨值曲線
            realized_pnl = sum(t.pnl for t in self.trade_history if t.pnl != 0)
            self.equity_curve.append((timestamp, price, equity, realized_pnl, unrealized_pnl))

            # 檢查最大回撤
            drawdown = 1 - (equity / self.max_equity) if self.max_equity > 0 else 0
            if drawdown >= self.config.max_drawdown:
                break

            final_price = price

        return self._generate_result(final_price)

    def _generate_result(self, final_price: float) -> BacktestResult:
        """生成回測結果"""
        # 計算盈虧
        unrealized_pnl = self._calculate_unrealized_pnl(final_price)
        realized_pnl = sum(t.pnl for t in self.trade_history if t.pnl != 0)
        final_equity = self.balance + unrealized_pnl

        # 計算勝率
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl < 0]
        total_closed = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_closed if total_closed > 0 else 0

        # 計算盈虧比
        total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 計算 Sharpe Ratio (修正版 - 根據實際數據時間跨度年化)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1][2]
                curr_equity = self.equity_curve[i][2]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)

            if returns and len(returns) > 1:
                import statistics
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)

                if std_return > 0:
                    # 計算實際時間跨度 (分鐘)
                    start_time = self.equity_curve[0][0]
                    end_time = self.equity_curve[-1][0]
                    if hasattr(start_time, 'timestamp'):
                        # datetime 對象
                        duration_minutes = (end_time - start_time).total_seconds() / 60
                    else:
                        # 假設是 timestamp (毫秒)
                        duration_minutes = (end_time - start_time) / 60000

                    # 計算每個週期的長度 (分鐘)
                    periods = len(returns)
                    minutes_per_period = duration_minutes / periods if periods > 0 else 1

                    # 年化因子: 一年有多少個這樣的週期
                    # 1年 = 365天 * 24小時 * 60分鐘 = 525,600 分鐘
                    periods_per_year = 525600 / minutes_per_period if minutes_per_period > 0 else 525600

                    # Sharpe Ratio = (平均收益 / 標準差) * sqrt(年化因子)
                    sharpe_ratio = (avg_return / std_return) * (periods_per_year ** 0.5)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return BacktestResult(
            final_equity=final_equity,
            return_pct=(final_equity - self.config.initial_balance) / self.config.initial_balance,
            max_drawdown=1 - final_equity / self.max_equity if self.max_equity > 0 else 0,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            trades_count=len(self.trade_history),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            direction=self.config.direction,
            config=self.config,
            trade_history=self.trade_history,
            equity_curve=self.equity_curve
        )

    def get_trade_df(self) -> pd.DataFrame:
        """取得交易記錄 DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "action": t.action,
                "price": t.price,
                "quantity": t.quantity,
                "side": t.side,
                "pnl": t.pnl,
                "fee": t.fee,
                "equity": t.equity
            }
            for t in self.trade_history
        ])

    def get_equity_df(self) -> pd.DataFrame:
        """取得淨值曲線 DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame(
            self.equity_curve,
            columns=["timestamp", "price", "equity", "realized_pnl", "unrealized_pnl"]
        )
