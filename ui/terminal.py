# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
終端 UI
=======
交易面板顯示
"""

from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from config.models import GlobalConfig, GlobalState

console = Console()


class TerminalUI:
    """終端 UI 顯示"""

    def __init__(self, config: GlobalConfig, state: GlobalState, bot=None):
        self.config = config
        self.state = state
        self.bot = bot

    def create_header(self) -> Panel:
        status = "[green]● 運行中[/]" if self.state.running else "[red]● 已停止[/]"
        ws_status = "[green]WS[/]" if self.state.connected else "[yellow]WS斷開[/]"
        if self.state.start_time:
            duration = datetime.now() - self.state.start_time
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            runtime = "--:--:--"
        enabled_count = len([s for s in self.config.symbols.values() if s.enabled])
        header = Text()
        header.append("AS 網格交易系統 ", style="bold cyan")
        header.append("MAX ", style="bold yellow")
        header.append(f"│ {status} │ {ws_status} ", style="")
        header.append(f"│ {enabled_count} 交易對 ", style="dim")
        header.append(f"│ {runtime}", style="dim")
        return Panel(header, box=box.DOUBLE_EDGE, style="cyan")

    def create_account_panel(self) -> Panel:
        table = Table(box=None, show_header=False, expand=True, padding=(0, 1))
        table.add_column("", style="dim", width=8)
        table.add_column("USDC", justify="right", width=12)
        table.add_column("USDT", justify="right", width=12)
        table.add_row("", "[cyan bold]USDC[/]", "[yellow bold]USDT[/]")
        usdc = self.state.get_account("USDC")
        usdt = self.state.get_account("USDT")
        table.add_row("權益",
            f"[white]{usdc.equity:.2f}[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[white]{usdt.equity:.2f}[/]" if usdt.equity > 0 else "[dim]--[/]")
        table.add_row("可用",
            f"{usdc.available_balance:.2f}" if usdc.available_balance > 0 else "[dim]--[/]",
            f"{usdt.available_balance:.2f}" if usdt.available_balance > 0 else "[dim]--[/]")
        def margin_style(ratio):
            return "green" if ratio < 0.3 else "yellow" if ratio < 0.6 else "red"
        table.add_row("保證金",
            f"[{margin_style(usdc.margin_ratio)}]{usdc.margin_ratio*100:.1f}%[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[{margin_style(usdt.margin_ratio)}]{usdt.margin_ratio*100:.1f}%[/]" if usdt.equity > 0 else "[dim]--[/]")
        def pnl_style(pnl):
            return "green" if pnl >= 0 else "red"
        table.add_row("浮盈",
            f"[{pnl_style(usdc.unrealized_pnl)}]{'+' if usdc.unrealized_pnl >= 0 else ''}{usdc.unrealized_pnl:.2f}[/]" if usdc.equity > 0 else "[dim]--[/]",
            f"[{pnl_style(usdt.unrealized_pnl)}]{'+' if usdt.unrealized_pnl >= 0 else ''}{usdt.unrealized_pnl:.2f}[/]" if usdt.equity > 0 else "[dim]--[/]")
        table.add_row("", "", "")
        pnl_color = "green" if self.state.total_unrealized_pnl >= 0 else "red"
        pnl_sign = "+" if self.state.total_unrealized_pnl >= 0 else ""
        table.add_row("[bold]總計[/]", f"[bold white]{self.state.total_equity:.2f}[/]", f"[{pnl_color}]{pnl_sign}{self.state.total_unrealized_pnl:.2f}[/]")
        return Panel(table, title="[bold]帳戶[/]", box=box.ROUNDED)

    def create_symbols_panel(self) -> Panel:
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("交易對", style="cyan")
        table.add_column("價格", justify="right")
        table.add_column("多", justify="right")
        table.add_column("空", justify="right")
        table.add_column("浮盈", justify="right")
        table.add_column("狀態", justify="center")
        for sym_config in self.config.symbols.values():
            if not sym_config.enabled:
                continue
            sym_state = self.state.symbols.get(sym_config.ccxt_symbol)
            if not sym_state:
                continue
            price_str = f"{sym_state.latest_price:.4f}" if sym_state.latest_price else "--"
            long_style = "green" if sym_state.long_position > 0 else "dim"
            short_style = "red" if sym_state.short_position > 0 else "dim"
            pnl = sym_state.unrealized_pnl
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""
            status_parts = []
            if sym_state.long_position > sym_config.position_threshold:
                status_parts.append("[red bold]多裝死[/]")
            elif sym_state.long_position > sym_config.position_limit:
                status_parts.append("[yellow]多×2[/]")
            if sym_state.short_position > sym_config.position_threshold:
                status_parts.append("[red bold]空裝死[/]")
            elif sym_state.short_position > sym_config.position_limit:
                status_parts.append("[yellow]空×2[/]")
            status_str = " ".join(status_parts) if status_parts else "[dim green]正常[/]"
            table.add_row(f"{sym_config.coin_name}", price_str,
                f"[{long_style}]{sym_state.long_position:.1f}[/]",
                f"[{short_style}]{sym_state.short_position:.1f}[/]",
                f"[{pnl_style}]{pnl_sign}{pnl:.2f}[/]", status_str)
        return Panel(table, title="[bold]交易對狀態[/]", box=box.ROUNDED)

    def create_max_panel(self) -> Panel:
        table = Table(box=None, show_header=False, expand=True)
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        bandit_cfg = self.config.bandit
        if bandit_cfg.enabled and self.bot:
            bandit = self.bot.bandit_optimizer
            params = bandit.get_current_params()
            table.add_row("學習次數", f"[green]#{bandit.total_pulls}[/]")
            table.add_row("當前間距", f"[cyan]{params.grid_spacing*100:.1f}%/{params.take_profit_spacing*100:.1f}%[/]")
            table.add_row("γ係數", f"{params.gamma:.2f}")
        else:
            table.add_row("Bandit", "[dim]OFF[/]")
        if self.config.max_enhancement.all_enhancements_enabled:
            table.add_row("增強模式", "[green]ON[/]")
        else:
            table.add_row("增強模式", "[dim]OFF[/]")
        return Panel(table, title="[bold yellow]AS 學習[/]", box=box.ROUNDED)

    def create_help_panel(self) -> Panel:
        help_text = Text()
        help_text.append("[Ctrl+C]", style="bold cyan")
        help_text.append(" 返回選單", style="dim")
        return Panel(help_text, box=box.ROUNDED, style="dim")

    def create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="left", size=38),
            Layout(name="symbols")
        )
        layout["left"].split_column(
            Layout(name="account"),
            Layout(name="max", size=10)
        )
        layout["header"].update(self.create_header())
        layout["account"].update(self.create_account_panel())
        layout["max"].update(self.create_max_panel())
        layout["symbols"].update(self.create_symbols_panel())
        layout["footer"].update(self.create_help_panel())
        return layout