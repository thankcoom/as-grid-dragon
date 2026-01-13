# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
領先指標管理器
=============
OFI、成交量、價差分析
"""

import time
import logging
from collections import deque
from typing import Optional, List, Dict, Tuple

import numpy as np

from config.models import LeadingIndicatorConfig

logger = logging.getLogger("as_grid_max")


class LeadingIndicatorManager:
    """
    領先指標管理器

    取代滯後的 ATR 波動率指標，使用三種領先因子：

    1. Order Flow Imbalance (OFI)
       公式: OFI = (買量 - 賣量) / (買量 + 賣量)
       範圍: -1 到 1
       - OFI > 0.6: 強烈買壓，價格可能上漲
       - OFI < -0.6: 強烈賣壓，價格可能下跌
       應用: 偏向調整方向

    2. Volume Surge (成交量突增)
       公式: Volume Ratio = 當前成交量 / 平均成交量
       信號: Ratio > 2.0 表示異常放量
       應用: 放大間距，避免被突破掃掉

    3. Spread Expansion (價差擴大)
       公式: Spread Ratio = 當前價差 / 平均價差
       信號: Ratio > 1.5 表示流動性下降
       應用: 預期波動放大，調整間距
    """

    def __init__(self, config: LeadingIndicatorConfig = None):
        self.config = config or LeadingIndicatorConfig()

        # 數據存儲 (每個交易對)
        self.trade_history: Dict[str, deque] = {}      # 成交記錄
        self.spread_history: Dict[str, deque] = {}     # 價差記錄
        self.ofi_history: Dict[str, deque] = {}        # OFI 歷史

        # 當前狀態
        self.current_ofi: Dict[str, float] = {}
        self.current_volume_ratio: Dict[str, float] = {}
        self.current_spread_ratio: Dict[str, float] = {}

        # 信號狀態
        self.active_signals: Dict[str, List[str]] = {}

        logger.info("[LeadingIndicator] 領先指標管理器初始化完成")

    def _ensure_symbol_data(self, symbol: str):
        """確保交易對數據結構存在"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=500)
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=200)
        if symbol not in self.ofi_history:
            self.ofi_history[symbol] = deque(maxlen=100)

    def record_trade(self, symbol: str, price: float, quantity: float, side: str):
        """
        記錄成交 (從 WebSocket 接收)

        Args:
            symbol: 交易對
            price: 成交價
            quantity: 成交量
            side: 'buy' 或 'sell'
        """
        if not self.config.enabled:
            return

        self._ensure_symbol_data(symbol)

        self.trade_history[symbol].append({
            'time': time.time(),
            'price': price,
            'quantity': quantity,
            'side': side,
            'value': price * quantity
        })

    def update_spread(self, symbol: str, bid: float, ask: float):
        """
        更新買賣價差

        Args:
            symbol: 交易對
            bid: 最佳買價
            ask: 最佳賣價
        """
        if not self.config.enabled or bid <= 0 or ask <= 0:
            return

        self._ensure_symbol_data(symbol)

        mid_price = (bid + ask) / 2
        spread_bps = (ask - bid) / mid_price * 10000  # 基點

        self.spread_history[symbol].append({
            'time': time.time(),
            'bid': bid,
            'ask': ask,
            'spread_bps': spread_bps
        })

    def calculate_ofi(self, symbol: str) -> float:
        """
        計算 Order Flow Imbalance

        Returns:
            OFI 值 (-1 到 1)
            - 正值: 買壓強
            - 負值: 賣壓強
            - 接近 0: 平衡
        """
        if symbol not in self.trade_history:
            return 0.0

        trades = list(self.trade_history[symbol])
        if len(trades) < self.config.ofi_lookback:
            return 0.0

        # 取最近 N 筆
        recent = trades[-self.config.ofi_lookback:]

        buy_volume = sum(t['value'] for t in recent if t['side'] == 'buy')
        sell_volume = sum(t['value'] for t in recent if t['side'] == 'sell')

        total = buy_volume + sell_volume
        if total <= 0:
            return 0.0

        ofi = (buy_volume - sell_volume) / total

        # 更新狀態
        self.current_ofi[symbol] = ofi
        self.ofi_history[symbol].append({
            'time': time.time(),
            'ofi': ofi,
            'buy_vol': buy_volume,
            'sell_vol': sell_volume
        })

        return ofi

    def calculate_volume_ratio(self, symbol: str) -> float:
        """
        計算成交量比率

        Returns:
            當前成交量 / 平均成交量
            - > 2.0: 異常放量
            - < 0.5: 異常縮量
        """
        if symbol not in self.trade_history:
            return 1.0

        trades = list(self.trade_history[symbol])
        if len(trades) < self.config.volume_lookback:
            return 1.0

        # 按分鐘聚合
        now = time.time()
        recent_minute = [t['value'] for t in trades if now - t['time'] < 60]
        historical = trades[-self.config.volume_lookback:]

        current_volume = sum(recent_minute)
        avg_volume_per_trade = np.mean([t['value'] for t in historical])
        expected_volume = avg_volume_per_trade * max(1, len(recent_minute))

        if expected_volume <= 0:
            return 1.0

        ratio = current_volume / expected_volume

        self.current_volume_ratio[symbol] = ratio
        return ratio

    def calculate_spread_ratio(self, symbol: str) -> float:
        """
        計算價差比率

        Returns:
            當前價差 / 平均價差
            - > 1.5: 價差擴大 (流動性下降)
            - < 0.8: 價差收窄 (流動性良好)
        """
        if symbol not in self.spread_history:
            return 1.0

        spreads = list(self.spread_history[symbol])
        if len(spreads) < self.config.spread_lookback:
            return 1.0

        current_spread = spreads[-1]['spread_bps']
        avg_spread = np.mean([s['spread_bps'] for s in spreads[-self.config.spread_lookback:]])

        if avg_spread <= 0:
            return 1.0

        ratio = current_spread / avg_spread

        self.current_spread_ratio[symbol] = ratio
        return ratio

    def get_signals(self, symbol: str) -> Tuple[List[str], Dict[str, float]]:
        """
        獲取當前活躍信號

        Returns:
            (信號列表, 指標值字典)
        """
        if not self.config.enabled:
            return [], {}

        signals = []
        values = {}

        # 計算各指標
        ofi = self.calculate_ofi(symbol)
        volume_ratio = self.calculate_volume_ratio(symbol)
        spread_ratio = self.calculate_spread_ratio(symbol)

        values = {
            'ofi': ofi,
            'volume_ratio': volume_ratio,
            'spread_ratio': spread_ratio
        }

        # OFI 信號
        if self.config.ofi_enabled:
            if ofi > self.config.ofi_threshold:
                signals.append('OFI_BUY_PRESSURE')
            elif ofi < -self.config.ofi_threshold:
                signals.append('OFI_SELL_PRESSURE')

        # Volume 信號
        if self.config.volume_enabled:
            if volume_ratio > self.config.volume_surge_threshold:
                signals.append('VOLUME_SURGE')

        # Spread 信號
        if self.config.spread_enabled:
            if spread_ratio > self.config.spread_surge_threshold:
                signals.append('SPREAD_EXPANSION')

        self.active_signals[symbol] = signals
        return signals, values

    def get_spacing_adjustment(self, symbol: str, base_spacing: float) -> Tuple[float, str]:
        """
        根據領先指標計算間距調整

        核心邏輯:
        - 正常情況: 保持基礎間距
        - Volume Surge 或 Spread Expansion: 放大間距 (預期波動)
        - OFI 極端值: 可能方向性移動，謹慎調整

        Args:
            symbol: 交易對
            base_spacing: 基礎間距

        Returns:
            (調整後間距, 原因說明)
        """
        if not self.config.enabled:
            return base_spacing, "領先指標關閉"

        signals, values = self.get_signals(symbol)

        # 沒有信號，保持原樣
        if not signals:
            return base_spacing, "正常"

        # 計算調整係數
        adjustment = 1.0
        reasons = []

        # Volume Surge: 放大間距 20-50%
        if 'VOLUME_SURGE' in signals:
            vol_ratio = values.get('volume_ratio', 1.0)
            # 成交量越大，間距放大越多
            vol_adj = min(1.5, 1.0 + (vol_ratio - 2.0) * 0.1)
            adjustment = max(adjustment, vol_adj)
            reasons.append(f"放量×{vol_ratio:.1f}")

        # Spread Expansion: 放大間距 20-40%
        if 'SPREAD_EXPANSION' in signals:
            spread_ratio = values.get('spread_ratio', 1.0)
            spread_adj = min(1.4, 1.0 + (spread_ratio - 1.5) * 0.2)
            adjustment = max(adjustment, spread_adj)
            reasons.append(f"價差擴{spread_ratio:.1f}x")

        # OFI 極端: 小幅放大間距 (防止被單邊掃)
        if 'OFI_BUY_PRESSURE' in signals or 'OFI_SELL_PRESSURE' in signals:
            ofi = abs(values.get('ofi', 0))
            ofi_adj = 1.0 + ofi * 0.2  # 最多放大 20%
            adjustment = max(adjustment, ofi_adj)
            direction = "買" if values.get('ofi', 0) > 0 else "賣"
            reasons.append(f"{direction}壓OFI={ofi:.2f}")

        # 限制最大調整
        adjustment = min(adjustment, 1.8)  # 最多放大 80%

        adjusted_spacing = base_spacing * adjustment
        reason = " + ".join(reasons) if reasons else "正常"

        return adjusted_spacing, reason

    def get_direction_bias(self, symbol: str) -> Tuple[float, float, str]:
        """
        根據 OFI 計算方向偏向

        當 OFI 顯示強烈方向時，調整多空偏好

        Returns:
            (long_bias, short_bias, reason)
            - bias > 1.0: 增加該方向
            - bias < 1.0: 減少該方向
        """
        if not self.config.enabled or not self.config.ofi_enabled:
            return 1.0, 1.0, ""

        ofi = self.current_ofi.get(symbol, 0)

        if abs(ofi) < self.config.ofi_threshold * 0.5:
            return 1.0, 1.0, "OFI平衡"

        # OFI > 0: 買壓強，偏向做多
        # OFI < 0: 賣壓強，偏向做空
        bias_strength = abs(ofi) * 0.3  # 最多 30% 偏向

        if ofi > 0:
            long_bias = 1.0 + bias_strength
            short_bias = 1.0 - bias_strength * 0.5
            reason = f"買壓+{ofi:.2f}"
        else:
            long_bias = 1.0 - bias_strength * 0.5
            short_bias = 1.0 + bias_strength
            reason = f"賣壓{ofi:.2f}"

        return long_bias, short_bias, reason

    def should_pause_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        判斷是否應該暫停交易

        極端情況:
        - Volume 異常 (> 4x) + Spread 異常 (> 2x) = 暫停
        - 可能是大消息或閃崩

        Returns:
            (是否暫停, 原因)
        """
        if not self.config.enabled:
            return False, ""

        signals, values = self.get_signals(symbol)

        volume_ratio = values.get('volume_ratio', 1.0)
        spread_ratio = values.get('spread_ratio', 1.0)

        # 極端條件
        if volume_ratio > 4.0 and spread_ratio > 2.0:
            return True, f"極端波動 (Vol={volume_ratio:.1f}x, Spread={spread_ratio:.1f}x)"

        # 單一極端
        if volume_ratio > 6.0:
            return True, f"異常放量 (Vol={volume_ratio:.1f}x)"

        if spread_ratio > 3.0:
            return True, f"流動性枯竭 (Spread={spread_ratio:.1f}x)"

        return False, ""

    def get_stats(self, symbol: str = None) -> Dict:
        """獲取統計資訊"""
        if symbol:
            signals, values = self.get_signals(symbol)
            return {
                'symbol': symbol,
                'enabled': self.config.enabled,
                'ofi': values.get('ofi', 0),
                'volume_ratio': values.get('volume_ratio', 1.0),
                'spread_ratio': values.get('spread_ratio', 1.0),
                'active_signals': signals,
                'trade_count': len(self.trade_history.get(symbol, [])),
                'spread_count': len(self.spread_history.get(symbol, []))
            }

        return {
            'enabled': self.config.enabled,
            'symbols': list(self.trade_history.keys()),
            'config': self.config.to_dict()
        }
