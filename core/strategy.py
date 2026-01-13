# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
網格策略核心邏輯
================
統一回測與實盤的策略計算
"""

from typing import Tuple


class GridStrategy:
    """
    網格策略核心邏輯 - 統一回測與實盤

    此類提取所有策略計算邏輯，確保回測與實盤行為一致。
    不包含任何 I/O 操作（下單、日誌等），只負責純計算。

    使用方式:
    - 回測: BacktestManager 調用靜態方法計算價格/數量
    - 實盤: MaxGridBot 調用相同方法，確保邏輯一致
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 常量定義 - 集中管理魔術數字
    # ═══════════════════════════════════════════════════════════════════════════
    DEAD_MODE_FALLBACK_LONG = 1.05    # 多頭裝死模式無對手倉時的止盈比例
    DEAD_MODE_FALLBACK_SHORT = 0.95   # 空頭裝死模式無對手倉時的止盈比例
    DEAD_MODE_DIVISOR = 100           # 裝死模式計算除數 (持倉比/100)

    @staticmethod
    def is_dead_mode(position: float, threshold: float) -> bool:
        """
        判斷是否進入裝死模式

        Args:
            position: 當前持倉量
            threshold: 裝死閾值 (position_threshold)

        Returns:
            True = 進入裝死模式，停止補倉
        """
        return position > threshold

    @staticmethod
    def calculate_dead_mode_price(
        base_price: float,
        my_position: float,
        opposite_position: float,
        side: str,
        fallback_long: float = None,
        fallback_short: float = None
    ) -> float:
        """
        計算裝死模式的特殊止盈價格

        公式:
        - 多頭: price × ((多倉/空倉)/100 + 1)，無空倉時 price × 1.05
        - 空頭: price ÷ ((空倉/多倉)/100 + 1)，無多倉時 price × 0.95

        設計理念:
        - 持倉比例越失衡，止盈價格越遠
        - 等待極端反彈才平倉，避免在不利位置出場

        Args:
            base_price: 基準價格 (當前價或上次網格價)
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            side: 'long' 或 'short'

        Returns:
            特殊止盈價格
        """
        if opposite_position > 0:
            r = (my_position / opposite_position) / GridStrategy.DEAD_MODE_DIVISOR + 1
            if side == 'long':
                return base_price * r
            else:  # short
                return base_price / r
        else:
            # 無對手倉，使用固定比例 (支持自定義 fallback)
            fb_long = fallback_long if fallback_long is not None else GridStrategy.DEAD_MODE_FALLBACK_LONG
            fb_short = fallback_short if fallback_short is not None else GridStrategy.DEAD_MODE_FALLBACK_SHORT
            if side == 'long':
                return base_price * fb_long
            else:  # short
                return base_price * fb_short

    @staticmethod
    def calculate_tp_quantity(
        base_qty: float,
        my_position: float,
        opposite_position: float,
        position_limit: float,
        position_threshold: float
    ) -> float:
        """
        計算止盈數量

        加倍條件 (滿足任一):
        1. 本方向持倉 > position_limit
        2. 對手方向持倉 >= position_threshold

        設計理念:
        - 持倉過大時加速出場
        - 對手進入裝死時也加速，維持多空平衡

        Args:
            base_qty: 基礎數量 (initial_quantity)
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            position_limit: 持倉上限 (觸發加倍)
            position_threshold: 裝死閾值 (對手觸發時也加倍)

        Returns:
            調整後的止盈數量 (1x 或 2x)
        """
        if my_position > position_limit or opposite_position >= position_threshold:
            return base_qty * 2
        return base_qty

    @staticmethod
    def calculate_grid_prices(
        base_price: float,
        take_profit_spacing: float,
        grid_spacing: float,
        side: str
    ) -> Tuple[float, float]:
        """
        計算正常模式的網格價格

        Args:
            base_price: 基準價格
            take_profit_spacing: 止盈間距 (如 0.004 = 0.4%)
            grid_spacing: 補倉間距 (如 0.006 = 0.6%)
            side: 'long' 或 'short'

        Returns:
            (止盈價格, 補倉價格)
        """
        if side == 'long':
            tp_price = base_price * (1 + take_profit_spacing)
            entry_price = base_price * (1 - grid_spacing)
        else:  # short
            tp_price = base_price * (1 - take_profit_spacing)
            entry_price = base_price * (1 + grid_spacing)

        return tp_price, entry_price

    @staticmethod
    def get_grid_decision(
        price: float,
        my_position: float,
        opposite_position: float,
        position_threshold: float,
        position_limit: float,
        base_qty: float,
        take_profit_spacing: float,
        grid_spacing: float,
        side: str,
        dead_mode_enabled: bool = True,
        fallback_long: float = None,
        fallback_short: float = None
    ) -> dict:
        """
        獲取完整的網格決策 (主要入口方法)

        統一回測與實盤的決策邏輯，返回所有需要的計算結果。
        此方法是無副作用的純函數，可用於回測和實盤。

        Args:
            price: 當前/基準價格
            my_position: 本方向持倉量
            opposite_position: 對手方向持倉量
            position_threshold: 裝死閾值
            position_limit: 持倉上限
            base_qty: 基礎數量
            take_profit_spacing: 止盈間距
            grid_spacing: 補倉間距
            side: 'long' 或 'short'

        Returns:
            {
                'dead_mode': bool,       # 是否裝死模式
                'tp_price': float,       # 止盈價格
                'entry_price': float,    # 補倉價格 (裝死模式為 None)
                'tp_qty': float,         # 止盈數量
                'entry_qty': float,      # 補倉數量 (裝死模式為 0)
            }

        Example:
            >>> decision = GridStrategy.get_grid_decision(
            ...     price=2.5, my_position=100, opposite_position=50,
            ...     position_threshold=500, position_limit=100,
            ...     base_qty=10, take_profit_spacing=0.004,
            ...     grid_spacing=0.006, side='long'
            ... )
            >>> decision['dead_mode']
            False
            >>> decision['tp_price']
            2.51  # 2.5 * 1.004
        """
        dead_mode = dead_mode_enabled and GridStrategy.is_dead_mode(my_position, position_threshold)

        tp_qty = GridStrategy.calculate_tp_quantity(
            base_qty, my_position, opposite_position,
            position_limit, position_threshold
        )

        if dead_mode:
            tp_price = GridStrategy.calculate_dead_mode_price(
                price, my_position, opposite_position, side,
                fallback_long, fallback_short
            )
            return {
                'dead_mode': True,
                'tp_price': tp_price,
                'entry_price': None,
                'tp_qty': tp_qty,
                'entry_qty': 0,
            }
        else:
            tp_price, entry_price = GridStrategy.calculate_grid_prices(
                price, take_profit_spacing, grid_spacing, side
            )
            return {
                'dead_mode': False,
                'tp_price': tp_price,
                'entry_price': entry_price,
                'tp_qty': tp_qty,
                'entry_qty': base_qty,
            }
