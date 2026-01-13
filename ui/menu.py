# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
主選單
======
交易系統主選單介面
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .terminal import TerminalUI
from config.models import GlobalConfig, SymbolConfig
from core.backtest import BacktestManager
from utils import normalize_symbol

console = Console()


class MainMenu:
    """主選單"""

    def __init__(self):
        self.config = GlobalConfig.load()
        self.backtest_manager = BacktestManager()
        self.bot = None
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_loop = None
        self._trading_active = False

    def show_banner(self):
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]AS 網格交易系統[/] [bold yellow]MAX[/]\n"
            "[dim]模組化重構版本 - AS刷怪籠[/]",
            border_style="yellow"
        ))
        console.print()

    def main_menu(self):
        while True:
            self.show_banner()
            if self._trading_active and self.bot:
                console.print("[bold green]● 交易運行中[/]", end="  ")
                if self.bot.state.start_time:
                    duration = datetime.now() - self.bot.state.start_time
                    hours, remainder = divmod(int(duration.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    console.print(f"[dim]運行時間: {hours:02d}:{minutes:02d}:{seconds:02d}[/]", end="  ")
                console.print(f"[dim]浮盈: {self.bot.state.total_unrealized_pnl:+.2f}[/]\n")
            if self.config.symbols:
                enabled = [s for s in self.config.symbols.values() if s.enabled]
                console.print(f"[dim]已配置 {len(self.config.symbols)} 個交易對，{len(enabled)} 個啟用[/]\n")
            console.print("[bold]請選擇操作:[/]\n")
            if self._trading_active:
                console.print("  [cyan]1[/] 查看交易面板")
                console.print("  [cyan]s[/] [red]停止交易[/]")
            else:
                console.print("  [cyan]1[/] 開始交易")
            console.print("  [cyan]2[/] 管理交易對")
            console.print("  [cyan]3[/] 回測/優化")
            console.print("  [cyan]4[/] MAX 增強設定")
            console.print("  [cyan]5[/] 學習模組設定")
            console.print("  [cyan]6[/] 風控設定")
            console.print("  [cyan]7[/] API 設定")
            console.print("  [cyan]0[/] 退出")
            console.print()
            valid_choices = ["0", "1", "2", "3", "4", "5", "6", "7"]
            if self._trading_active:
                valid_choices.append("s")
            choice = Prompt.ask("選擇", choices=valid_choices, default="1")

            # 使用 dispatch 字典簡化選單邏輯
            menu_actions = {
                "2": self.manage_symbols,
                "3": self.quick_backtest,
                "4": self.setup_max_enhancement,
                "5": self.setup_learning,
                "6": self.setup_risk,
                "7": self.setup_api,
            }

            if choice == "0":
                if self._handle_exit():
                    break
            elif choice == "1":
                self._handle_trading_toggle()
            elif choice == "s" and self._trading_active:
                self.stop_trading()
            elif choice in menu_actions:
                menu_actions[choice]()

    def _handle_exit(self) -> bool:
        """處理退出選項。回傳 True 表示應該退出。"""
        if self._trading_active:
            if Confirm.ask("[yellow]交易運行中，確定要退出嗎？[/]"):
                self.stop_trading()
                return True
            return False
        return True

    def _handle_trading_toggle(self):
        """處理交易切換 (選項 1)。"""
        if self._trading_active:
            self.view_trading_panel()
        else:
            self.start_trading()

    def start_trading(self):
        if not self.config.api_key:
            console.print("[red]請先設定 API[/]")
            Prompt.ask("按 Enter 繼續")
            return
        enabled = [s for s in self.config.symbols.values() if s.enabled]
        if not enabled:
            console.print("[red]沒有啟用的交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return
        console.print("[bold]啟動 MAX 網格交易...[/]\n")
        from core.bot import MaxGridBot
        self.bot = MaxGridBot(self.config)
        def run_bot_thread():
            self.bot_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.bot_loop)
            try:
                self.bot_loop.run_until_complete(self.bot.run())
            except Exception as e:
                from constants import logger
                logger.error(f"Bot 運行錯誤: {e}")
            finally:
                self.bot_loop.close()
                self._trading_active = False
        self.bot_thread = threading.Thread(target=run_bot_thread, daemon=True)
        self.bot_thread.start()
        with console.status("[bold cyan]連接交易所...[/]"):
            for _ in range(100):
                if self.bot.state.running:
                    break
                time.sleep(0.1)
        if self.bot.state.running:
            self._trading_active = True
            console.print("[bold green]✓ 交易已在背景啟動！[/]\n")
        else:
            console.print("[red]Bot 啟動失敗[/]")
            self.bot = None
        Prompt.ask("按 Enter 繼續")

    def stop_trading(self):
        if not self._trading_active or not self.bot:
            return
        console.print("[bold yellow]正在停止交易...[/]")
        if self.bot_loop and self.bot_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.bot.stop(), self.bot_loop)
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=5)
        self._trading_active = False
        self.bot = None
        console.print("[green]✓ 交易已停止[/]")
        Prompt.ask("按 Enter 繼續")

    def view_trading_panel(self):
        if not self._trading_active or not self.bot:
            return
        ui = TerminalUI(self.config, self.bot.state, self.bot)
        console.print("[dim]按 Ctrl+C 返回主選單[/]\n")
        try:
            with Live(ui.create_layout(), console=console, refresh_per_second=2) as live:
                while self._trading_active and self.bot.state.running:
                    live.update(ui.create_layout())
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def manage_symbols(self):
        while True:
            self.show_banner()
            console.print("[bold]交易對管理[/]\n")
            if self.config.symbols:
                table = Table(box=box.ROUNDED)
                table.add_column("#", style="dim")
                table.add_column("交易對", style="cyan")
                table.add_column("狀態")
                table.add_column("止盈", justify="right")
                table.add_column("補倉", justify="right")
                table.add_column("數量", justify="right")
                for i, cfg in enumerate(self.config.symbols.values(), 1):
                    status = "[green]啟用[/]" if cfg.enabled else "[dim]停用[/]"
                    table.add_row(str(i), cfg.symbol, status,
                        f"{cfg.take_profit_spacing*100:.2f}%", f"{cfg.grid_spacing*100:.2f}%",
                        str(cfg.initial_quantity))
                console.print(table)
                console.print()
            console.print("  [cyan]a[/] 新增交易對")
            console.print("  [cyan]e[/] 編輯交易對")
            console.print("  [cyan]d[/] 刪除交易對")
            console.print("  [cyan]t[/] 切換啟用/停用")
            console.print("  [cyan]0[/] 返回")
            console.print()
            choice = Prompt.ask("選擇", choices=["0", "a", "e", "d", "t"], default="0")
            if choice == "0":
                break
            elif choice == "a":
                self.add_symbol()
            elif choice == "e":
                self.edit_symbol()
            elif choice == "d":
                self.delete_symbol()
            elif choice == "t":
                self.toggle_symbol()

    def add_symbol(self):
        symbol_input = Prompt.ask("輸入交易對 (如 XRPUSDC)")
        raw, ccxt, coin, quote = normalize_symbol(symbol_input)
        if not raw:
            console.print("[red]無法識別的交易對格式[/]")
            Prompt.ask("按 Enter 繼續")
            return
        if raw in self.config.symbols:
            console.print(f"[yellow]{raw} 已存在[/]")
            Prompt.ask("按 Enter 繼續")
            return
        take_profit = FloatPrompt.ask("止盈間距 (%)", default=0.4) / 100
        grid_spacing = FloatPrompt.ask("補倉間距 (%)", default=0.6) / 100
        quantity = FloatPrompt.ask("每單數量", default=3.0)
        leverage = IntPrompt.ask("槓桿", default=20)
        self.config.symbols[raw] = SymbolConfig(
            symbol=raw, ccxt_symbol=ccxt, enabled=True,
            take_profit_spacing=take_profit, grid_spacing=grid_spacing,
            initial_quantity=quantity, leverage=leverage)
        self.config.save()
        console.print(f"[green]已新增 {raw}[/]")
        Prompt.ask("按 Enter 繼續")

    def edit_symbol(self):
        if not self.config.symbols:
            console.print("[yellow]沒有可編輯的交易對[/]")
            Prompt.ask("按 Enter 繼續")
            return
        symbols = list(self.config.symbols.keys())
        idx = IntPrompt.ask("輸入序號編輯", default=1) - 1
        if idx < 0 or idx >= len(symbols):
            console.print("[red]無效序號[/]")
            Prompt.ask("按 Enter 繼續")
            return
        key = symbols[idx]
        cfg = self.config.symbols[key]
        console.print(f"\n[bold cyan]編輯 {key}[/]\n")
        # 顯示當前計算值
        console.print(f"[dim]當前 position_limit = {cfg.position_limit:.1f} (數量×{cfg.limit_multiplier})[/]")
        console.print(f"[dim]當前 position_threshold = {cfg.position_threshold:.1f} (數量×{cfg.threshold_multiplier})[/]\n")
        # 基礎參數
        cfg.take_profit_spacing = FloatPrompt.ask(f"止盈間距 (%)", default=cfg.take_profit_spacing * 100) / 100
        cfg.grid_spacing = FloatPrompt.ask(f"補倉間距 (%)", default=cfg.grid_spacing * 100) / 100
        cfg.initial_quantity = FloatPrompt.ask(f"每單數量", default=cfg.initial_quantity)
        cfg.leverage = IntPrompt.ask(f"槓桿", default=cfg.leverage)
        # 進階參數
        if Confirm.ask("調整進階參數 (倍數控制)?", default=False):
            cfg.limit_multiplier = FloatPrompt.ask(f"止盈加倍倍數", default=cfg.limit_multiplier)
            cfg.threshold_multiplier = FloatPrompt.ask(f"裝死倍數", default=cfg.threshold_multiplier)
        console.print(f"\n[dim]新 position_limit = {cfg.position_limit:.1f}[/]")
        console.print(f"[dim]新 position_threshold = {cfg.position_threshold:.1f}[/]")
        self.config.save()
        console.print("\n[green]已更新[/]")
        Prompt.ask("按 Enter 繼續")

    def delete_symbol(self):
        if not self.config.symbols:
            return
        symbols = list(self.config.symbols.keys())
        idx = IntPrompt.ask("輸入序號刪除", default=1) - 1
        if idx < 0 or idx >= len(symbols):
            return
        key = symbols[idx]
        if Confirm.ask(f"確定刪除 {key}?"):
            del self.config.symbols[key]
            self.config.save()
            console.print("[green]已刪除[/]")
        Prompt.ask("按 Enter 繼續")

    def toggle_symbol(self):
        if not self.config.symbols:
            return
        symbols = list(self.config.symbols.keys())
        idx = IntPrompt.ask("輸入序號切換", default=1) - 1
        if idx < 0 or idx >= len(symbols):
            return
        key = symbols[idx]
        self.config.symbols[key].enabled = not self.config.symbols[key].enabled
        self.config.save()
        status = "啟用" if self.config.symbols[key].enabled else "停用"
        console.print(f"[green]{key} 已{status}[/]")
        Prompt.ask("按 Enter 繼續")

    def quick_backtest(self):
        self.show_banner()
        console.print("[bold]回測/優化[/]\n")
        symbol_input = Prompt.ask("交易對").strip()
        raw, ccxt_sym, coin, quote = normalize_symbol(symbol_input)
        if not raw:
            console.print(f"[red]無法識別交易對: {symbol_input}[/]")
            Prompt.ask("按 Enter 繼續")
            return
        console.print(f"\n[green]識別為: {coin}/{quote} ({raw})[/]\n")
        # 選擇模式
        console.print("  [cyan]1[/] 單筆回測 (使用當前參數)")
        console.print("  [cyan]2[/] 參數優化 (測試多組參數)")
        mode = Prompt.ask("選擇", choices=["1", "2"], default="1")
        # 選擇日期
        today = datetime.now()
        console.print("\n  [cyan]a[/] 最近 7 天")
        console.print("  [cyan]b[/] 最近 30 天")
        console.print("  [cyan]c[/] 最近 90 天")
        date_choice = Prompt.ask("日期範圍", choices=["a", "b", "c"], default="b")
        days = {"a": 7, "b": 30, "c": 90}[date_choice]
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        # 載入數據
        available_dates = self.backtest_manager.get_available_dates(raw)
        need_download = any(
            (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d") not in available_dates
            for i in range(days))
        if need_download:
            console.print("\n[yellow]下載數據中...[/]\n")
            self.backtest_manager.download_data(raw, ccxt_sym, start_date, end_date)
        df = self.backtest_manager.load_data(raw, start_date, end_date)
        if df is None or df.empty:
            console.print("[red]載入數據失敗[/]")
            Prompt.ask("按 Enter 繼續")
            return
        console.print(f"[green]載入 {len(df):,} 條 K 線 ({start_date} ~ {end_date})[/]\n")
        sym_config = self.config.symbols.get(raw) or SymbolConfig(symbol=raw, ccxt_symbol=ccxt_sym)
        if mode == "1":
            # 單筆回測模式
            self._run_single_backtest(sym_config, df, raw)
        else:
            # 參數優化模式
            self._run_optimize(sym_config, df, raw)
        Prompt.ask("\n按 Enter 繼續")

    def _run_single_backtest(self, sym_config: SymbolConfig, df, raw: str):
        """單筆回測 - 詳細報告"""
        console.print(f"[cyan]使用當前參數回測...[/]")
        console.print(f"  止盈: {sym_config.take_profit_spacing*100:.2f}%")
        console.print(f"  補倉: {sym_config.grid_spacing*100:.2f}%")
        console.print(f"  數量: {sym_config.initial_quantity}\n")
        result = self.backtest_manager.run_backtest(sym_config, df)
        # 詳細報告
        console.print("[bold cyan]═══════════════ 回測報告 ═══════════════[/]\n")
        ret_color = "green" if result['return_pct'] >= 0 else "red"
        pnl_color = "green" if result['realized_pnl'] >= 0 else "red"
        console.print(f"  [bold]收益概況[/]")
        console.print(f"    初始資金: {result.get('initial_balance', 1000):.2f} U")
        console.print(f"    最終淨值: [{ret_color}]{result['final_equity']:.2f} U[/]")
        console.print(f"    收益率:   [{ret_color}]{result['return_pct']*100:+.2f}%[/]")
        console.print(f"    最大回撤: [yellow]{result['max_drawdown']*100:.2f}%[/]\n")
        console.print(f"  [bold]交易統計[/]")
        console.print(f"    總交易數: {result['trades_count']}")
        console.print(f"    多單成交: {result.get('long_trades', 0)}")
        console.print(f"    空單成交: {result.get('short_trades', 0)}")
        console.print(f"    勝率:     {result['win_rate']*100:.1f}%")
        pf = result['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        console.print(f"    盈虧比:   {pf_str}\n")
        console.print(f"  [bold]盈虧詳情[/]")
        console.print(f"    已實現: [{pnl_color}]{result['realized_pnl']:+.2f} U[/]")
        console.print(f"    未實現: {result['unrealized_pnl']:+.2f} U")
        avg_win = result.get('avg_win', 0)
        avg_loss = result.get('avg_loss', 0)
        if avg_win or avg_loss:
            console.print(f"    平均獲利: [green]{avg_win:+.2f} U[/]")
            console.print(f"    平均虧損: [red]{avg_loss:+.2f} U[/]")
        # 簡易收益曲線
        equity_curve = result.get('equity_curve', [])
        if equity_curve and len(equity_curve) > 10:
            console.print("\n  [bold]收益走勢[/] (簡化)")
            samples = [equity_curve[i * len(equity_curve) // 10] for i in range(10)] + [equity_curve[-1]]
            min_eq, max_eq = min(samples), max(samples)
            bar_width = 30
            for i, eq in enumerate(samples):
                pct = int((eq - min_eq) / (max_eq - min_eq + 0.01) * bar_width)
                bar = "█" * pct + "░" * (bar_width - pct)
                color = "green" if eq >= 1000 else "red"
                console.print(f"    [{color}]{bar}[/] {eq:.1f}")
        console.print("\n[bold cyan]════════════════════════════════════════[/]")

    def _run_optimize(self, sym_config: SymbolConfig, df, raw: str):
        """參數優化模式"""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("優化中...", total=100)
            def update_progress(current, total):
                progress.update(task, completed=current * 100 // total)
            results = self.backtest_manager.optimize_params(sym_config, df, update_progress)
        console.print("\n[bold cyan]優化結果 (Top 5)[/]\n")
        table = Table(box=box.ROUNDED)
        table.add_column("#")
        table.add_column("止盈")
        table.add_column("補倉")
        table.add_column("收益率")
        table.add_column("回撤")
        table.add_column("勝率")
        table.add_column("交易數")
        for i, r in enumerate(results[:5], 1):
            color = "green" if r["return_pct"] >= 0 else "red"
            table.add_row(str(i), f"{r['take_profit_spacing']*100:.2f}%", f"{r['grid_spacing']*100:.2f}%",
                f"[{color}]{r['return_pct']*100:.2f}%[/]", f"{r['max_drawdown']*100:.1f}%",
                f"{r['win_rate']*100:.1f}%", str(r['trades_count']))
        console.print(table)
        if results and Confirm.ask("\n是否應用最佳參數?"):
            best = results[0]
            if raw not in self.config.symbols:
                self.config.symbols[raw] = sym_config
            self.config.symbols[raw].take_profit_spacing = best["take_profit_spacing"]
            self.config.symbols[raw].grid_spacing = best["grid_spacing"]
            self.config.save()
            console.print(f"[green]已應用: 止盈 {best['take_profit_spacing']*100:.2f}%, 補倉 {best['grid_spacing']*100:.2f}%[/]")

    def setup_max_enhancement(self):
        self.show_banner()
        console.print("[bold yellow]MAX 增強功能設定[/]\n")
        max_cfg = self.config.max_enhancement
        mode_status = "[bold green]增強模式[/]" if max_cfg.all_enhancements_enabled else "[bold cyan]純淨模式[/]"
        console.print(f"[bold]當前模式:[/] {mode_status}\n")
        if Confirm.ask("是否修改設定?"):
            max_cfg.all_enhancements_enabled = Confirm.ask("啟用增強模式?", default=max_cfg.all_enhancements_enabled)
            if max_cfg.all_enhancements_enabled:
                max_cfg.funding_rate_enabled = Confirm.ask("啟用 Funding Rate 偏向?", default=max_cfg.funding_rate_enabled)
                max_cfg.glft_enabled = Confirm.ask("啟用 GLFT 庫存控制?", default=max_cfg.glft_enabled)
                max_cfg.dynamic_grid_enabled = Confirm.ask("啟用動態網格?", default=max_cfg.dynamic_grid_enabled)
            self.config.save()
            console.print("[green]設定已保存[/]")
        Prompt.ask("按 Enter 繼續")

    def setup_learning(self):
        self.show_banner()
        console.print("[bold yellow]學習模組設定[/]\n")
        bandit = self.config.bandit
        leading = self.config.leading_indicator
        # 顯示當前狀態
        console.print("[bold]━━━ UCB Bandit 參數學習 ━━━[/]")
        status = "[green]啟用[/]" if bandit.enabled else "[dim]停用[/]"
        console.print(f"  狀態: {status}")
        if bandit.enabled:
            console.print(f"  [dim]探索係數: {bandit.exploration_factor}[/]")
            console.print(f"  [dim]窗口大小: {bandit.window_size}[/]")
            console.print(f"  [dim]Contextual: {'是' if bandit.contextual_enabled else '否'}[/]")
            console.print(f"  [dim]Thompson: {'是' if bandit.thompson_enabled else '否'}[/]")
            # 如果有運行中的 bot，顯示學習統計
            if self._trading_active and self.bot:
                stats = self.bot.bandit_optimizer.get_stats()
                console.print(f"\n  [cyan]學習統計:[/]")
                console.print(f"    總學習次數: [yellow]{stats['total_pulls']}[/]")
                console.print(f"    當前參數: {stats['current_arm']}")
                console.print(f"    最佳參數: [green]{stats['best_arm']}[/]")
                console.print(f"    累積收益: {stats['cumulative_reward']:.2f}")
                if stats.get('current_context'):
                    console.print(f"    市場狀態: {stats['current_context']}")
        console.print()
        console.print("[bold]━━━ 領先指標系統 ━━━[/]")
        status = "[green]啟用[/]" if leading.enabled else "[dim]停用[/]"
        console.print(f"  狀態: {status}")
        if leading.enabled:
            console.print(f"  [dim]OFI 閾值: {leading.ofi_threshold}[/]")
            console.print(f"  [dim]成交量閾值: {leading.volume_surge_threshold}x[/]")
            console.print(f"  [dim]價差閾值: {leading.spread_surge_threshold}x[/]")
        console.print()
        # 修改選項
        console.print("  [cyan]1[/] 切換 Bandit 開關")
        console.print("  [cyan]2[/] 切換領先指標開關")
        console.print("  [cyan]3[/] 調整 Bandit 參數")
        console.print("  [cyan]4[/] 調整領先指標參數")
        console.print("  [cyan]0[/] 返回")
        console.print()
        choice = Prompt.ask("選擇", choices=["0", "1", "2", "3", "4"], default="0")
        if choice == "1":
            bandit.enabled = not bandit.enabled
            self.config.save()
            console.print(f"[green]Bandit 已{'啟用' if bandit.enabled else '停用'}[/]")
        elif choice == "2":
            leading.enabled = not leading.enabled
            self.config.save()
            console.print(f"[green]領先指標已{'啟用' if leading.enabled else '停用'}[/]")
        elif choice == "3":
            console.print("\n[bold]Bandit 參數調整[/]")
            bandit.exploration_factor = FloatPrompt.ask("探索係數 (越大越愛探索)", default=bandit.exploration_factor)
            bandit.window_size = IntPrompt.ask("滑動窗口大小", default=bandit.window_size)
            bandit.contextual_enabled = Confirm.ask("啟用 Contextual (市場狀態感知)?", default=bandit.contextual_enabled)
            bandit.thompson_enabled = Confirm.ask("啟用 Thompson Sampling?", default=bandit.thompson_enabled)
            self.config.save()
            console.print("[green]Bandit 參數已更新[/]")
        elif choice == "4":
            console.print("\n[bold]領先指標參數調整[/]")
            leading.ofi_threshold = FloatPrompt.ask("OFI 閾值 (0-1)", default=leading.ofi_threshold)
            leading.volume_surge_threshold = FloatPrompt.ask("成交量突增閾值 (倍數)", default=leading.volume_surge_threshold)
            leading.spread_surge_threshold = FloatPrompt.ask("價差擴大閾值 (倍數)", default=leading.spread_surge_threshold)
            self.config.save()
            console.print("[green]領先指標參數已更新[/]")
        Prompt.ask("按 Enter 繼續")

    def setup_risk(self):
        self.show_banner()
        console.print("[bold]風控設定[/]\n")
        risk = self.config.risk
        console.print(f"  狀態: {'[green]啟用[/]' if risk.enabled else '[dim]停用[/]'}")
        console.print(f"  保證金閾值: {risk.margin_threshold*100:.0f}%")
        console.print(f"  啟動追蹤: 浮盈 >= {risk.trailing_start_profit:.1f}U")
        console.print()
        if Confirm.ask("是否修改設定?"):
            risk.enabled = Confirm.ask("啟用追蹤止盈?", default=risk.enabled)
            if risk.enabled:
                risk.margin_threshold = FloatPrompt.ask(f"保證金閾值 (%)", default=risk.margin_threshold * 100) / 100
                risk.trailing_start_profit = FloatPrompt.ask(f"啟動追蹤閾值 (U)", default=risk.trailing_start_profit)
            self.config.save()
            console.print("[green]設定已保存[/]")
        Prompt.ask("按 Enter 繼續")

    def setup_api(self):
        self.show_banner()
        console.print("[bold]API 設定[/]\n")
        if self.config.api_key:
            console.print(f"[dim]當前 API Key: {self.config.api_key[:8]}...{self.config.api_key[-4:]}[/]")
            # 測試現有 API
            if Confirm.ask("測試現有 API 連線?", default=True):
                self._test_api_connection()
            if not Confirm.ask("是否重新設定 API?"):
                return
        api_key = Prompt.ask("API Key")
        api_secret = Prompt.ask("API Secret")
        # 測試新 API
        console.print("\n[cyan]測試 API 連線...[/]")
        if self._test_api_connection(api_key, api_secret):
            self.config.api_key = api_key
            self.config.api_secret = api_secret
            self.config.save()
            console.print("[green]API 設定完成並已驗證[/]")
        else:
            if Confirm.ask("[yellow]API 驗證失敗，是否仍要保存?[/]", default=False):
                self.config.api_key = api_key
                self.config.api_secret = api_secret
                self.config.save()
        Prompt.ask("按 Enter 繼續")

    def _test_api_connection(self, api_key: str = None, api_secret: str = None) -> bool:
        """測試 API 連線"""
        import ccxt
        key = api_key or self.config.api_key
        secret = api_secret or self.config.api_secret
        if not key or not secret:
            console.print("[red]未設定 API[/]")
            return False
        try:
            exchange = ccxt.binance({
                "apiKey": key,
                "secret": secret,
                "options": {"defaultType": "future"}
            })
            balance = exchange.fetch_balance()
            # 顯示餘額
            console.print("\n[bold green]✓ API 連線成功[/]\n")
            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("幣種", style="cyan")
            table.add_column("可用", justify="right")
            table.add_column("總計", justify="right")
            for currency in ["USDC", "USDT", "BNB"]:
                if currency in balance:
                    info = balance[currency]
                    free = float(info.get('free', 0) or 0)
                    total = float(info.get('total', 0) or 0)
                    if total > 0:
                        table.add_row(currency, f"{free:.4f}", f"{total:.4f}")
            console.print(table)
            # 檢查期貨權限
            try:
                exchange.fetch_positions()
                console.print("[green]✓ 期貨交易權限正常[/]")
            except Exception:
                console.print("[yellow]⚠ 無法讀取期貨倉位 (可能需要開通期貨權限)[/]")
            return True
        except ccxt.AuthenticationError:
            console.print("[red]✗ API 驗證失敗 (Key 或 Secret 錯誤)[/]")
            return False
        except ccxt.PermissionDenied:
            console.print("[red]✗ API 權限不足[/]")
            return False
        except Exception as e:
            console.print(f"[red]✗ 連線錯誤: {e}[/]")
            return False
