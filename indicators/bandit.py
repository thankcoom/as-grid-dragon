# Author: louis
# Threads: https://www.threads.com/@mr.__.l
"""
UCB Bandit 參數優化器
====================
基於 TradeBot 論文，自動學習最佳參數組合
"""

import time
import logging
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Dict

import numpy as np

from config.models import BanditConfig

logger = logging.getLogger("as_grid_max")


# ═══════════════════════════════════════════════════════════════════════════
# 市場狀態枚舉 (用於 Contextual Bandit)
# ═══════════════════════════════════════════════════════════════════════════
class MarketContext:
    """
    市場狀態分類

    用於 Contextual Bandit，根據市場狀態選擇不同的參數策略:
    - RANGING: 震盪市 → 緊密間距，高頻交易
    - TRENDING_UP: 上漲趨勢 → 偏多策略
    - TRENDING_DOWN: 下跌趨勢 → 偏空策略
    - HIGH_VOLATILITY: 高波動 → 寬鬆間距，避免被掃
    """
    RANGING = "ranging"              # 震盪
    TRENDING_UP = "trending_up"      # 上漲趨勢
    TRENDING_DOWN = "trending_down"  # 下跌趨勢
    HIGH_VOLATILITY = "high_vol"     # 高波動

    # 每種市場狀態的推薦 arm 索引
    RECOMMENDED_ARMS = {
        RANGING: [0, 1, 2, 3],        # 緊密型，適合震盪
        TRENDING_UP: [4, 5],           # 平衡型
        TRENDING_DOWN: [4, 5],         # 平衡型
        HIGH_VOLATILITY: [6, 7, 8, 9]  # 寬鬆型，適合高波動
    }


@dataclass
class ParameterArm:
    """參數組合 (一個 Arm)"""
    gamma: float                # GLFT 風險係數
    grid_spacing: float         # 補倉間距
    take_profit_spacing: float  # 止盈間距

    def __hash__(self):
        return hash((self.gamma, self.grid_spacing, self.take_profit_spacing))

    def __str__(self):
        return f"γ={self.gamma:.2f}/GS={self.grid_spacing*100:.1f}%/TP={self.take_profit_spacing*100:.1f}%"


class UCBBanditOptimizer:
    """
    UCB Bandit 參數優化器 (增強版)

    基於論文: TradeBot: Bandit learning for hyper-parameters optimization

    增強功能:
    1. 冷啟動預載 - 首次運行使用歷史最佳參數，避免初期隨機探索的損失
    2. Contextual Bandit - 根據市場狀態 (趨勢/震盪/高波動) 選擇不同策略
    3. Thompson Sampling - 支持連續參數空間探索，通過參數擾動發現更優解
    4. 改進 Reward - 加入 Max Drawdown 懲罰和勝率獎勵

    AS 網格特點:
    - 每次只掛一買一賣
    - 成交後跟隨價格重新掛單
    - 對向倍投 (持倉多時反向加倍)
    """

    # AS 網格專用參數組合空間
    DEFAULT_ARMS = [
        # 緊密型 (高頻交易，手續費敏感) - 適合震盪市
        ParameterArm(gamma=0.05, grid_spacing=0.003, take_profit_spacing=0.003),
        ParameterArm(gamma=0.05, grid_spacing=0.004, take_profit_spacing=0.004),
        # 非對稱型 (止盈小於補倉，適合震盪)
        ParameterArm(gamma=0.08, grid_spacing=0.005, take_profit_spacing=0.003),
        ParameterArm(gamma=0.08, grid_spacing=0.006, take_profit_spacing=0.004),
        # 平衡型 - 適合趨勢市
        ParameterArm(gamma=0.10, grid_spacing=0.006, take_profit_spacing=0.004),
        ParameterArm(gamma=0.10, grid_spacing=0.008, take_profit_spacing=0.005),
        # 寬鬆型 (低頻交易) - 適合高波動
        ParameterArm(gamma=0.12, grid_spacing=0.008, take_profit_spacing=0.006),
        ParameterArm(gamma=0.12, grid_spacing=0.010, take_profit_spacing=0.006),
        # 高波動適應型
        ParameterArm(gamma=0.15, grid_spacing=0.010, take_profit_spacing=0.008),
        ParameterArm(gamma=0.15, grid_spacing=0.012, take_profit_spacing=0.008),
    ]

    def __init__(self, config: BanditConfig = None):
        self.config = config or BanditConfig()
        self.arms = self.DEFAULT_ARMS.copy()

        # 每個 arm 的獎勵歷史 (滑動窗口)
        self.rewards: Dict[int, deque] = {
            i: deque(maxlen=self.config.window_size)
            for i in range(len(self.arms))
        }

        # 追蹤狀態
        self.current_arm_idx: int = 0
        self.total_pulls: int = 0
        self.pull_counts: Dict[int, int] = {i: 0 for i in range(len(self.arms))}

        # 交易追蹤 (用於計算 reward)
        self.pending_trades: List[Dict] = []
        self.trade_count_since_update: int = 0

        # 統計
        self.best_arm_history: List[int] = []
        self.cumulative_reward: float = 0

        # === 新增: Contextual Bandit 狀態 ===
        self.current_context: str = MarketContext.RANGING
        self.price_history: deque = deque(maxlen=100)  # 價格歷史 (用於計算市場狀態)

        # 每個 context 的獨立統計
        self.context_rewards: Dict[str, Dict[int, deque]] = {
            ctx: {i: deque(maxlen=self.config.window_size) for i in range(len(self.arms))}
            for ctx in [MarketContext.RANGING, MarketContext.TRENDING_UP,
                       MarketContext.TRENDING_DOWN, MarketContext.HIGH_VOLATILITY]
        }
        self.context_pulls: Dict[str, Dict[int, int]] = {
            ctx: {i: 0 for i in range(len(self.arms))}
            for ctx in [MarketContext.RANGING, MarketContext.TRENDING_UP,
                       MarketContext.TRENDING_DOWN, MarketContext.HIGH_VOLATILITY]
        }

        # === 新增: Thompson Sampling 狀態 ===
        # Beta 分布參數 (alpha, beta) 用於每個 arm
        self.thompson_alpha: Dict[int, float] = {
            i: self.config.thompson_prior_alpha for i in range(len(self.arms))
        }
        self.thompson_beta: Dict[int, float] = {
            i: self.config.thompson_prior_beta for i in range(len(self.arms))
        }

        # === 新增: 動態生成的 arm (Thompson Sampling 探索) ===
        self.dynamic_arm: Optional[ParameterArm] = None
        self.dynamic_arm_reward: float = 0

        # === 冷啟動初始化 ===
        if self.config.cold_start_enabled:
            self._cold_start_init()

        logger.info(f"[Bandit] 增強版初始化完成，共 {len(self.arms)} 個參數組合")
        logger.info(f"[Bandit] 功能: 冷啟動={self.config.cold_start_enabled}, "
                   f"Contextual={self.config.contextual_enabled}, "
                   f"Thompson={self.config.thompson_enabled}")

    def _cold_start_init(self):
        """
        冷啟動初始化

        設計理念:
        - 首次運行時，避免從頭探索造成的損失
        - 使用歷史回測得出的最佳參數作為起點
        - 為推薦的 arm 預設一些正向獎勵
        """
        # 設定初始 arm 為歷史最佳
        self.current_arm_idx = self.config.cold_start_arm_idx

        # 為推薦 arm 預載一些虛擬正向獎勵 (給予信任)
        recommended_arms = [4, 5]  # 平衡型
        for arm_idx in recommended_arms:
            self.rewards[arm_idx].append(0.5)  # 預設正向獎勵
            self.pull_counts[arm_idx] = 1
            self.total_pulls += 1

        logger.info(f"[Bandit] 冷啟動: 初始 arm={self.current_arm_idx}, "
                   f"預載 arms={recommended_arms}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Contextual Bandit: 市場狀態檢測
    # ═══════════════════════════════════════════════════════════════════════════

    def update_price(self, price: float):
        """更新價格歷史 (用於計算市場狀態)"""
        self.price_history.append(price)

    def detect_market_context(self) -> str:
        """
        檢測當前市場狀態

        基於:
        1. 波動率 (標準差 / 均值)
        2. 趨勢 (線性回歸斜率)

        Returns:
            MarketContext 枚舉值
        """
        if not self.config.contextual_enabled:
            return MarketContext.RANGING

        if len(self.price_history) < self.config.volatility_lookback:
            return self.current_context  # 數據不足，保持當前狀態

        prices = list(self.price_history)

        # 計算波動率
        recent_prices = prices[-self.config.volatility_lookback:]
        volatility = np.std(recent_prices) / np.mean(recent_prices)

        # 檢查高波動
        if volatility > self.config.high_volatility_threshold:
            self.current_context = MarketContext.HIGH_VOLATILITY
            return self.current_context

        # 計算趨勢 (簡單線性回歸)
        if len(prices) >= self.config.trend_lookback:
            trend_prices = prices[-self.config.trend_lookback:]
            x = np.arange(len(trend_prices))
            slope = np.polyfit(x, trend_prices, 1)[0]
            trend_pct = slope / np.mean(trend_prices)

            if trend_pct > self.config.trend_threshold:
                self.current_context = MarketContext.TRENDING_UP
            elif trend_pct < -self.config.trend_threshold:
                self.current_context = MarketContext.TRENDING_DOWN
            else:
                self.current_context = MarketContext.RANGING
        else:
            self.current_context = MarketContext.RANGING

        return self.current_context

    # ═══════════════════════════════════════════════════════════════════════════
    # Thompson Sampling: 連續空間探索
    # ═══════════════════════════════════════════════════════════════════════════

    def _thompson_sample(self) -> int:
        """
        Thompson Sampling 選擇 arm

        從每個 arm 的 Beta 分布中採樣，選擇採樣值最高的 arm
        """
        samples = []
        for i in range(len(self.arms)):
            # 從 Beta 分布採樣
            sample = np.random.beta(self.thompson_alpha[i], self.thompson_beta[i])
            samples.append(sample)

        return int(np.argmax(samples))

    def _generate_dynamic_arm(self) -> Optional[ParameterArm]:
        """
        基於最佳 arm 生成動態參數組合 (參數擾動)

        設計理念:
        - 找到目前最佳的 arm
        - 在其參數附近進行小幅擾動
        - 探索可能更優的連續參數空間
        """
        if not self.config.thompson_enabled:
            return None

        # 找到最佳 arm
        best_idx = self._get_best_arm()
        best_arm = self.arms[best_idx]

        # 參數擾動
        perturbation = self.config.param_perturbation

        # 隨機擾動方向
        gamma_delta = np.random.uniform(-perturbation, perturbation) * best_arm.gamma
        gs_delta = np.random.uniform(-perturbation, perturbation) * best_arm.grid_spacing
        tp_delta = np.random.uniform(-perturbation, perturbation) * best_arm.take_profit_spacing

        # 生成新參數 (確保在合理範圍內)
        new_gamma = max(0.01, min(0.3, best_arm.gamma + gamma_delta))
        new_gs = max(0.002, min(0.02, best_arm.grid_spacing + gs_delta))
        new_tp = max(0.002, min(0.015, best_arm.take_profit_spacing + tp_delta))

        # 確保 tp < gs
        if new_tp >= new_gs:
            new_tp = new_gs * 0.7

        return ParameterArm(gamma=new_gamma, grid_spacing=new_gs, take_profit_spacing=new_tp)

    # ═══════════════════════════════════════════════════════════════════════════
    # 改進的 Reward 計算
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_reward(self, pnls: List[float]) -> float:
        """
        計算改進的 Reward

        公式:
        reward = sharpe_ratio - mdd_penalty * max_drawdown + win_rate_bonus * win_rate

        改進點:
        1. Sharpe-like 指標 (平均收益 / 波動)
        2. Max Drawdown 懲罰 (避免大回撤)
        3. 勝率獎勵 (鼓勵穩定獲利)
        """
        if not pnls:
            return 0

        # 1. Sharpe-like reward
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if np.std(pnls) > 0 else 0.001
        sharpe = mean_pnl / std_pnl

        # 2. Max Drawdown 計算
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # 正規化 MDD (相對於總收益)
        total_pnl = sum(pnls)
        mdd_ratio = max_drawdown / abs(total_pnl) if total_pnl != 0 else 0
        mdd_penalty = self.config.mdd_penalty_weight * mdd_ratio

        # 3. 勝率獎勵
        win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
        win_bonus = self.config.win_rate_bonus * (win_rate - 0.5)  # 50% 為基準

        # 綜合 reward
        reward = sharpe - mdd_penalty + win_bonus

        return reward

    def _update_thompson(self, reward: float):
        """
        更新 Thompson Sampling 的 Beta 分布參數

        將 reward 轉換為 0-1 之間的成功/失敗概率
        """
        arm_idx = self.current_arm_idx

        # 將 reward 正規化到 [0, 1]
        # 使用 sigmoid 轉換
        prob_success = 1 / (1 + np.exp(-reward))

        # 更新 Beta 分布參數
        self.thompson_alpha[arm_idx] += prob_success
        self.thompson_beta[arm_idx] += (1 - prob_success)

    # ═══════════════════════════════════════════════════════════════════════════
    # 核心方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_current_params(self) -> ParameterArm:
        """獲取當前選擇的參數"""
        # 如果有動態 arm 且表現良好，使用它
        if self.dynamic_arm and self.dynamic_arm_reward > 0:
            return self.dynamic_arm
        return self.arms[self.current_arm_idx]

    def select_arm(self) -> int:
        """
        選擇 arm (融合 UCB + Contextual + Thompson)

        優先順序:
        1. Contextual: 根據市場狀態限制候選 arms
        2. Thompson: 有一定概率使用 Thompson Sampling
        3. UCB: 默認使用 UCB 算法
        """
        # 確保每個 arm 都至少被試過 min_pulls_per_arm 次
        for i in range(len(self.arms)):
            if self.pull_counts[i] < self.config.min_pulls_per_arm:
                return i

        # Contextual: 獲取當前市場狀態推薦的 arms
        if self.config.contextual_enabled:
            context = self.detect_market_context()
            recommended = MarketContext.RECOMMENDED_ARMS.get(context, list(range(len(self.arms))))
        else:
            recommended = list(range(len(self.arms)))

        # Thompson Sampling: 有 30% 概率使用
        if self.config.thompson_enabled and np.random.random() < 0.3:
            thompson_choice = self._thompson_sample()
            # 如果 Thompson 選擇在推薦列表中，使用它
            if thompson_choice in recommended:
                return thompson_choice

        # UCB 算法 (只在推薦的 arms 中選擇)
        ucb_values = []
        for i in range(len(self.arms)):
            if i not in recommended:
                ucb_values.append(float('-inf'))
                continue

            rewards = list(self.rewards[i])
            if not rewards:
                ucb_values.append(float('inf'))
                continue

            mean_reward = np.mean(rewards)
            confidence = self.config.exploration_factor * np.sqrt(
                2 * np.log(self.total_pulls + 1) / len(rewards)
            )
            ucb_values.append(mean_reward + confidence)

        return int(np.argmax(ucb_values))

    def record_trade(self, pnl: float, side: str):
        """
        記錄交易結果

        Args:
            pnl: 交易盈虧
            side: 'long' 或 'short'
        """
        if not self.config.enabled:
            return

        self.pending_trades.append({
            'pnl': pnl,
            'side': side,
            'arm_idx': self.current_arm_idx,
            'context': self.current_context,
            'timestamp': time.time()
        })
        self.trade_count_since_update += 1

        # 每 N 筆交易更新一次
        if self.trade_count_since_update >= self.config.update_interval:
            self._update_and_select()

    def _update_and_select(self):
        """更新獎勵並選擇新的 arm"""
        if not self.pending_trades:
            return

        pnls = [t['pnl'] for t in self.pending_trades]

        # 使用改進的 reward 計算
        reward = self._calculate_reward(pnls)

        # 更新當前 arm 的獎勵
        arm_idx = self.current_arm_idx
        self.rewards[arm_idx].append(reward)
        self.pull_counts[arm_idx] += 1
        self.total_pulls += 1
        self.cumulative_reward += sum(pnls)

        # 更新 Contextual 統計
        if self.config.contextual_enabled:
            context = self.pending_trades[0].get('context', MarketContext.RANGING)
            self.context_rewards[context][arm_idx].append(reward)
            self.context_pulls[context][arm_idx] += 1

        # 更新 Thompson Sampling 參數
        if self.config.thompson_enabled:
            self._update_thompson(reward)

        # 選擇下一個 arm
        new_arm_idx = self.select_arm()
        if new_arm_idx != self.current_arm_idx:
            old_params = self.arms[self.current_arm_idx]
            new_params = self.arms[new_arm_idx]
            logger.info(f"[Bandit] 切換參數: {old_params} → {new_params} "
                       f"(context={self.current_context})")
            self.current_arm_idx = new_arm_idx

        # 偶爾嘗試動態生成的 arm (10% 概率)
        if self.config.thompson_enabled and np.random.random() < 0.1:
            self.dynamic_arm = self._generate_dynamic_arm()
            if self.dynamic_arm:
                logger.info(f"[Bandit] 動態探索: {self.dynamic_arm}")

        # 記錄最佳 arm
        self.best_arm_history.append(self._get_best_arm())

        # 清空待處理交易
        self.pending_trades = []
        self.trade_count_since_update = 0

    def _get_best_arm(self) -> int:
        """獲取目前表現最好的 arm"""
        best_idx = 0
        best_mean = float('-inf')

        for i in range(len(self.arms)):
            rewards = list(self.rewards[i])
            if rewards:
                mean = np.mean(rewards)
                if mean > best_mean:
                    best_mean = mean
                    best_idx = i

        return best_idx

    def get_stats(self) -> Dict:
        """獲取優化器統計 (增強版)"""
        best_idx = self._get_best_arm()
        arm_stats = []

        for i in range(len(self.arms)):
            rewards = list(self.rewards[i])
            arm_stats.append({
                'arm': str(self.arms[i]),
                'pulls': self.pull_counts[i],
                'mean_reward': np.mean(rewards) if rewards else 0,
                'is_current': i == self.current_arm_idx,
                'is_best': i == best_idx,
                'thompson_alpha': self.thompson_alpha[i],
                'thompson_beta': self.thompson_beta[i]
            })

        return {
            'enabled': self.config.enabled,
            'total_pulls': self.total_pulls,
            'current_arm': str(self.arms[self.current_arm_idx]),
            'best_arm': str(self.arms[best_idx]),
            'cumulative_reward': self.cumulative_reward,
            'current_context': self.current_context,
            'dynamic_arm': str(self.dynamic_arm) if self.dynamic_arm else None,
            'arm_stats': arm_stats
        }

    def to_dict(self) -> dict:
        """序列化狀態 (增強版)"""
        return {
            'current_arm_idx': self.current_arm_idx,
            'total_pulls': self.total_pulls,
            'pull_counts': dict(self.pull_counts),
            'rewards': {k: list(v) for k, v in self.rewards.items()},
            'cumulative_reward': self.cumulative_reward,
            'current_context': self.current_context,
            'thompson_alpha': dict(self.thompson_alpha),
            'thompson_beta': dict(self.thompson_beta),
            'context_pulls': {ctx: dict(pulls) for ctx, pulls in self.context_pulls.items()}
        }

    def load_state(self, state: dict):
        """載入狀態 (增強版)"""
        if not state:
            return
        self.current_arm_idx = state.get('current_arm_idx', 0)
        self.total_pulls = state.get('total_pulls', 0)
        self.pull_counts = {int(k): v for k, v in state.get('pull_counts', {}).items()}
        self.cumulative_reward = state.get('cumulative_reward', 0)
        self.current_context = state.get('current_context', MarketContext.RANGING)

        # 載入 rewards
        saved_rewards = state.get('rewards', {})
        for k, v in saved_rewards.items():
            idx = int(k)
            if idx in self.rewards:
                self.rewards[idx] = deque(v, maxlen=self.config.window_size)

        # 載入 Thompson 參數
        saved_alpha = state.get('thompson_alpha', {})
        for k, v in saved_alpha.items():
            self.thompson_alpha[int(k)] = v

        saved_beta = state.get('thompson_beta', {})
        for k, v in saved_beta.items():
            self.thompson_beta[int(k)] = v

        # 載入 context 統計
        saved_context_pulls = state.get('context_pulls', {})
        for ctx, pulls in saved_context_pulls.items():
            if ctx in self.context_pulls:
                self.context_pulls[ctx] = {int(k): v for k, v in pulls.items()}
