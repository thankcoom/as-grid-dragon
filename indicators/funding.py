# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
Funding Rate 與 GLFT 控制器
===========================
"""

import time
import logging
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config.models import MaxEnhancement

logger = logging.getLogger("as_grid_max")


class FundingRateManager:
    """Funding Rate 管理器"""

    def __init__(self, exchange):
        self.exchange = exchange
        self.funding_rates: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self.update_interval = 60

    def update_funding_rate(self, symbol: str) -> float:
        now = time.time()
        if symbol in self.last_update:
            if now - self.last_update[symbol] < self.update_interval:
                return self.funding_rates.get(symbol, 0)
        try:
            funding_info = self.exchange.fetch_funding_rate(symbol)
            rate = float(funding_info.get('fundingRate', 0) or 0)
            self.funding_rates[symbol] = rate
            self.last_update[symbol] = now
            logger.info(f"[Funding] {symbol} funding rate: {rate*100:.4f}%")
            return rate
        except Exception as e:
            logger.error(f"[Funding] 獲取 {symbol} funding rate 失敗: {e}")
            return self.funding_rates.get(symbol, 0)

    def get_position_bias(self, symbol: str, config: 'MaxEnhancement') -> Tuple[float, float]:
        if not config.is_feature_enabled('funding_rate'):
            return 1.0, 1.0
        rate = self.funding_rates.get(symbol, 0)
        if abs(rate) < config.funding_rate_threshold:
            return 1.0, 1.0
        bias = config.funding_rate_position_bias
        if rate > 0:
            return 1.0 - bias, 1.0 + bias
        else:
            return 1.0 + bias, 1.0 - bias


class GLFTController:
    """GLFT 庫存控制器"""

    def calculate_inventory_ratio(self, long_pos: float, short_pos: float) -> float:
        total = long_pos + short_pos
        if total <= 0:
            return 0.0
        return (long_pos - short_pos) / total

    def calculate_spread_skew(
        self, long_pos: float, short_pos: float, base_spread: float, config: 'MaxEnhancement'
    ) -> Tuple[float, float]:
        if not config.is_feature_enabled('glft'):
            return 0.0, 0.0
        inventory_ratio = self.calculate_inventory_ratio(long_pos, short_pos)
        skew = inventory_ratio * base_spread * config.gamma
        return -skew, skew

    def adjust_order_quantity(
        self, base_qty: float, side: str, long_pos: float, short_pos: float, config: 'MaxEnhancement'
    ) -> float:
        if not config.is_feature_enabled('glft'):
            return base_qty
        inventory_ratio = self.calculate_inventory_ratio(long_pos, short_pos)
        if side == 'long':
            adjust = 1.0 - inventory_ratio * config.gamma
        else:
            adjust = 1.0 + inventory_ratio * config.gamma
        return base_qty * max(0.5, min(1.5, adjust))
