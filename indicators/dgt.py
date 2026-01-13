# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
DGT 動態邊界與網格管理
======================
"""

import time
import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.models import DGTConfig, MaxEnhancement

logger = logging.getLogger("as_grid_max")


class DGTBoundaryManager:
    """DGT 動態邊界管理器"""

    def __init__(self, config: 'DGTConfig' = None):
        from config.models import DGTConfig
        self.config = config or DGTConfig()
        self.boundaries: Dict[str, Dict] = {}
        self.accumulated_profits: Dict[str, float] = {}
        self.reset_counts: Dict[str, int] = {}

    def initialize_boundary(self, symbol: str, center_price: float, grid_spacing: float, num_grids: int = 10):
        half_grids = num_grids // 2
        upper = center_price * ((1 + grid_spacing) ** half_grids)
        lower = center_price * ((1 - grid_spacing) ** half_grids)
        self.boundaries[symbol] = {
            'center': center_price, 'upper': upper, 'lower': lower,
            'grid_spacing': grid_spacing, 'num_grids': num_grids,
            'initialized_at': time.time(), 'last_reset': time.time()
        }
        self.accumulated_profits[symbol] = 0
        self.reset_counts[symbol] = 0
        logger.info(f"[DGT] {symbol} 邊界初始化: {lower:.4f} ~ {upper:.4f}")

    def check_and_reset(self, symbol: str, current_price: float, realized_pnl: float = 0) -> Tuple[bool, Optional[Dict]]:
        if not self.config.enabled or symbol not in self.boundaries:
            return False, None
        boundary = self.boundaries[symbol]
        breach_upper = current_price >= boundary['upper'] * (1 - self.config.boundary_buffer)
        breach_lower = current_price <= boundary['lower'] * (1 + self.config.boundary_buffer)
        if not (breach_upper or breach_lower):
            return False, None
        self.accumulated_profits[symbol] += realized_pnl
        old_center = boundary['center']
        reinvest = self.accumulated_profits[symbol] * (self.config.profit_reinvest_ratio if breach_upper else 1.0)
        self.initialize_boundary(symbol, current_price, boundary['grid_spacing'], boundary['num_grids'])
        self.reset_counts[symbol] += 1
        self.accumulated_profits[symbol] = 0
        return True, {'old_center': old_center, 'new_center': current_price,
                      'direction': 'upper' if breach_upper else 'lower',
                      'reinvest_amount': reinvest, 'reset_count': self.reset_counts[symbol]}

    def get_boundary_info(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.boundaries:
            return None
        b = self.boundaries[symbol]
        return {'center': b['center'], 'upper': b['upper'], 'lower': b['lower'],
                'reset_count': self.reset_counts.get(symbol, 0),
                'accumulated_profit': self.accumulated_profits.get(symbol, 0)}

    def get_stats(self) -> Dict:
        return {'enabled': self.config.enabled, 'symbols': list(self.boundaries.keys()),
                'total_resets': sum(self.reset_counts.values()),
                'boundaries': {s: self.get_boundary_info(s) for s in self.boundaries}}


class DynamicGridManager:
    """動態網格管理器 (ATR)"""

    def __init__(self):
        from collections import deque
        self.price_history: Dict[str, deque] = {}
        self.atr_cache: Dict[str, float] = {}
        self.last_calc_time: Dict[str, float] = {}
        self.calc_interval = 60

    def update_price(self, symbol: str, price: float):
        from collections import deque
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)
        self.price_history[symbol].append({'price': price, 'time': time.time()})

    def calculate_atr(self, symbol: str, config: 'MaxEnhancement') -> float:
        now = time.time()
        if symbol in self.last_calc_time and now - self.last_calc_time[symbol] < self.calc_interval:
            return self.atr_cache.get(symbol, 0.005)
        history = self.price_history.get(symbol, [])
        if len(history) < config.volatility_lookback:
            return 0.005
        recent_prices = [h['price'] for h in list(history)[-config.volatility_lookback:]]
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                   for i in range(1, len(recent_prices)) if recent_prices[i-1] > 0]
        if not returns:
            return 0.005
        volatility = np.std(returns) * config.atr_multiplier
        volatility = max(config.min_spacing, min(config.max_spacing, volatility))
        self.atr_cache[symbol] = volatility
        self.last_calc_time[symbol] = now
        return volatility

    def get_dynamic_spacing(self, symbol: str, base_tp: float, base_gs: float, config: 'MaxEnhancement') -> Tuple[float, float]:
        if not config.is_feature_enabled('dynamic_grid'):
            return base_tp, base_gs
        atr = self.calculate_atr(symbol, config)
        dynamic_tp = max(config.min_spacing, min(config.max_spacing * 0.6, atr * 0.5))
        dynamic_gs = max(config.min_spacing * 1.5, min(config.max_spacing, atr))
        if dynamic_tp >= dynamic_gs:
            dynamic_tp = dynamic_gs * 0.6
        return dynamic_tp, dynamic_gs
