# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
回測管理器
==========
歷史數據載入、回測執行、參數優化
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import pandas as pd

import ccxt

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .strategy import GridStrategy
from config.models import SymbolConfig
from constants import DATA_DIR

console = Console()


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

    def _create_exchange(self, exchange_type: str):
        """
        根據交易所類型創建 ccxt exchange 實例

        Args:
            exchange_type: 交易所類型 (binance, bybit, bitget, gate, okx)

        Returns:
            ccxt exchange 實例或 None
        """
        exchange_configs = {
            "binance": {
                "class": ccxt.binance,
                "options": {"defaultType": "future"}
            },
            "bybit": {
                "class": ccxt.bybit,
                "options": {"defaultType": "linear"}  # USDT 永續合約
            },
            "bitget": {
                "class": ccxt.bitget,
                "options": {"defaultType": "swap"}  # 永續合約
            },
            "gate": {
                "class": ccxt.gateio,
                "options": {"defaultType": "swap"}
            },
        }

        config = exchange_configs.get(exchange_type.lower())
        if config is None:
            return None

        try:
            return config["class"]({
                "enableRateLimit": True,
                "options": config["options"]
            })
        except Exception as e:
            console.print(f"[red]創建 {exchange_type} 交易所實例失敗: {e}[/]")
            return None

    def download_data(self, symbol_raw: str, ccxt_symbol: str, start_date: str, end_date: str,
                      exchange_type: str = "binance") -> bool:
        """
        下載歷史數據

        Args:
            symbol_raw: 原始交易對符號 (如 "XRPUSDC")
            ccxt_symbol: CCXT 格式符號 (如 "XRP/USDC:USDC")
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            exchange_type: 交易所類型 (binance, bybit, bitget, gate)
        """
        try:
            # 根據交易所類型創建 exchange 實例
            exchange = self._create_exchange(exchange_type)
            if exchange is None:
                console.print(f"[red]不支援的交易所: {exchange_type}[/]")
                return False

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

        # 持倉控制參數
        position_threshold = config.position_threshold
        position_limit = config.position_limit

        # 追蹤上次開倉價格 (用於計算止盈價)
        last_long_entry_price = df['close'].iloc[0]
        last_short_entry_price = df['close'].iloc[0]

        for _, row in df.iterrows():
            price = row['close']

            # 計算當前持倉量
            long_position = sum(p["qty"] for p in long_positions)
            short_position = sum(p["qty"] for p in short_positions)

            # === 同步實盤邏輯：使用當前價格計算網格 ===
            # 多頭網格
            long_decision = GridStrategy.get_grid_decision(
                price=price,  # 使用當前價格（同步實盤）
                my_position=long_position,
                opposite_position=short_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='long'
            )

            # 空頭網格
            short_decision = GridStrategy.get_grid_decision(
                price=price,  # 使用當前價格（同步實盤）
                my_position=short_position,
                opposite_position=long_position,
                position_threshold=position_threshold,
                position_limit=position_limit,
                base_qty=config.initial_quantity,
                take_profit_spacing=config.take_profit_spacing,
                grid_spacing=config.grid_spacing,
                side='short'
            )

            long_dead_mode = long_decision['dead_mode']
            short_dead_mode = short_decision['dead_mode']

            # === 多頭開倉邏輯 ===
            if not long_dead_mode:
                buy_price = long_decision['entry_price'] if long_decision['entry_price'] else price * (1 - config.grid_spacing)
                if price <= buy_price:
                    qty = order_value / price
                    margin = (qty * price) / leverage
                    fee = qty * price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        long_positions.append({"price": price, "qty": qty, "margin": margin})
                        last_long_entry_price = price

            # === 多頭止盈邏輯 ===
            if long_positions:
                # 止盈價格基於最早持倉的開倉價
                oldest_long_price = long_positions[0]["price"]
                sell_price = oldest_long_price * (1 + config.take_profit_spacing)
                long_tp_qty = long_decision['tp_qty']

                if price >= sell_price:
                    remaining_tp = long_tp_qty
                    while long_positions and remaining_tp > 0:
                        pos = long_positions[0]
                        if pos["qty"] <= remaining_tp:
                            long_positions.pop(0)
                            gross_pnl = (price - pos["price"]) * pos["qty"]
                            fee = pos["qty"] * price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += pos["margin"] + net_pnl
                            trades.append({"pnl": net_pnl, "type": "long"})
                            remaining_tp -= pos["qty"]
                        else:
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

            # === 空頭開倉邏輯 ===
            if not short_dead_mode:
                sell_short_price = short_decision['entry_price'] if short_decision['entry_price'] else price * (1 + config.grid_spacing)
                if price >= sell_short_price:
                    qty = order_value / price
                    margin = (qty * price) / leverage
                    fee = qty * price * fee_pct

                    if margin + fee < balance:
                        balance -= (margin + fee)
                        short_positions.append({"price": price, "qty": qty, "margin": margin})
                        last_short_entry_price = price

            # === 空頭止盈邏輯 ===
            if short_positions:
                # 止盈價格基於最早持倉的開倉價
                oldest_short_price = short_positions[0]["price"]
                cover_price = oldest_short_price * (1 - config.take_profit_spacing)
                short_tp_qty = short_decision['tp_qty']

                if price <= cover_price:
                    remaining_tp = short_tp_qty
                    while short_positions and remaining_tp > 0:
                        pos = short_positions[0]
                        if pos["qty"] <= remaining_tp:
                            short_positions.pop(0)
                            gross_pnl = (pos["price"] - price) * pos["qty"]
                            fee = pos["qty"] * price * fee_pct
                            net_pnl = gross_pnl - fee
                            balance += pos["margin"] + net_pnl
                            trades.append({"pnl": net_pnl, "type": "short"})
                            remaining_tp -= pos["qty"]
                        else:
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
            "initial_balance": 1000.0,
            "return_pct": (final_equity - 1000) / 1000,
            "max_drawdown": 1 - (min(equity_curve) / max_equity) if equity_curve else 0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "trades_count": len(trades),
            "win_rate": len(winning) / len(trades) if trades else 0,
            "profit_factor": sum(t["pnl"] for t in winning) / abs(sum(t["pnl"] for t in losing)) if losing else float('inf'),
            # 新增詳細統計
            "long_trades": len([t for t in trades if t["type"] == "long"]),
            "short_trades": len([t for t in trades if t["type"] == "short"]),
            "avg_win": sum(t["pnl"] for t in winning) / len(winning) if winning else 0,
            "avg_loss": sum(t["pnl"] for t in losing) / len(losing) if losing else 0,
            "equity_curve": equity_curve,  # 用於繪製收益曲線
            "max_equity": max_equity,
            "min_equity": min(equity_curve) if equity_curve else 1000.0,
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
