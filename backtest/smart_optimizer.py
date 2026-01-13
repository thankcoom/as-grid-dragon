"""
智能參數優化器模組
===================

基於 Optuna 框架的先進優化系統：
- TPE (Tree-structured Parzen Estimator) 貝葉斯優化
- 多目標優化 (Sharpe, Sortino, MaxDrawdown)
- 參數重要性分析
- 早期停止策略
- 可視化歷史追蹤

參考:
- Optuna: https://optuna.org
- Freqtrade Hyperopt
- 網格交易論文優化技術
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import logging
import time
import json
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler, NSGAIIISampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from .config import Config
from .backtester import GridBacktester, BacktestResult


class OptimizationObjective(Enum):
    """優化目標"""
    RETURN = "return"           # 收益率
    SHARPE = "sharpe"           # Sharpe Ratio
    SORTINO = "sortino"         # Sortino Ratio (只計算下行風險)
    CALMAR = "calmar"           # Calmar Ratio (收益/最大回撤)
    PROFIT_FACTOR = "profit_factor"  # 盈虧比
    RISK_ADJUSTED = "risk_adjusted"  # 風險調整收益 (收益 - 2*回撤)
    MULTI_OBJECTIVE = "multi"   # 多目標 (Pareto 優化)


class OptimizationMethod(Enum):
    """優化方法"""
    GRID_SEARCH = "grid"        # 網格搜索 (舊方法)
    TPE = "tpe"                 # Tree-structured Parzen Estimator
    CMA_ES = "cma_es"           # CMA-ES 演化策略
    NSGA_II = "nsga_ii"         # 多目標演化算法 NSGA-II
    NSGA_III = "nsga_iii"       # 多目標演化算法 NSGA-III


@dataclass
class TrialResult:
    """單次試驗結果"""
    trial_number: int
    params: Dict[str, float]
    metrics: Dict[str, float]
    objective_value: float
    duration: float

    def to_dict(self) -> dict:
        return {
            "trial": self.trial_number,
            **{f"param_{k}": v for k, v in self.params.items()},
            **self.metrics,
            "objective": self.objective_value,
            "duration_s": self.duration
        }


@dataclass
class SmartOptimizationResult:
    """智能優化結果"""
    best_params: Dict[str, float]
    best_metrics: Dict[str, float]
    best_objective: float
    all_trials: List[TrialResult]
    param_importance: Dict[str, float]
    pareto_front: Optional[List[Dict]] = None  # 多目標優化的 Pareto 前沿
    convergence_history: List[float] = field(default_factory=list)
    optimization_time: float = 0.0
    n_trials: int = 0
    method: str = "tpe"
    objective_type: str = "sharpe"

    def to_dataframe(self) -> pd.DataFrame:
        """轉換為 DataFrame"""
        rows = [t.to_dict() for t in self.all_trials]
        return pd.DataFrame(rows)

    def get_top_n(self, n: int = 5) -> pd.DataFrame:
        """獲取 Top N 結果"""
        df = self.to_dataframe()
        return df.nlargest(n, 'objective')

    def __str__(self) -> str:
        return (
            f"智能優化結果\n"
            f"{'='*50}\n"
            f"方法: {self.method.upper()}\n"
            f"目標: {self.objective_type}\n"
            f"試驗數: {self.n_trials}\n"
            f"耗時: {self.optimization_time:.1f}s\n"
            f"\n最佳參數:\n"
            f"  止盈間距: {self.best_params.get('take_profit_spacing', 0)*100:.3f}%\n"
            f"  補倉間距: {self.best_params.get('grid_spacing', 0)*100:.3f}%\n"
            f"  槓桿: {self.best_params.get('leverage', 20)}x\n"
            f"\n最佳績效:\n"
            f"  目標值: {self.best_objective:.4f}\n"
            f"  收益率: {self.best_metrics.get('return_pct', 0)*100:.2f}%\n"
            f"  Sharpe: {self.best_metrics.get('sharpe_ratio', 0):.3f}\n"
            f"  最大回撤: {self.best_metrics.get('max_drawdown', 0)*100:.2f}%\n"
            f"  勝率: {self.best_metrics.get('win_rate', 0)*100:.1f}%\n"
            f"\n參數重要性:\n" +
            "\n".join([f"  {k}: {v*100:.1f}%" for k, v in
                      sorted(self.param_importance.items(), key=lambda x: -x[1])])
        )


class SmartOptimizer:
    """
    智能參數優化器

    使用 Optuna 框架實現:
    - 貝葉斯優化 (TPE)
    - 多目標優化 (NSGA-II/III)
    - 參數重要性分析
    - 早期停止

    優勢:
    - 比網格搜索快 5-10 倍
    - 自動探索最有潛力的參數區域
    - 支持多目標權衡
    """

    # 參數邊界定義
    DEFAULT_PARAM_BOUNDS = {
        "take_profit_spacing": (0.001, 0.015),  # 0.1% ~ 1.5%
        "grid_spacing": (0.002, 0.025),          # 0.2% ~ 2.5%
        "leverage": (5, 50),                      # 5x ~ 50x
    }

    # 固定參數 (不優化)
    DEFAULT_FIXED_PARAMS = {
        "position_threshold": 500.0,
        "position_limit": 100.0,
        "max_positions": 50,
        "max_drawdown": 0.5,
        "fee_pct": 0.0004,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        base_config: Config = None,
        param_bounds: Dict[str, Tuple[float, float]] = None,
        fixed_params: Dict[str, Any] = None,
        logger: logging.Logger = None
    ):
        """
        初始化優化器

        Args:
            df: K線數據
            base_config: 基礎配置
            param_bounds: 參數範圍 {name: (min, max)}
            fixed_params: 固定參數
            logger: 日誌器
        """
        self.df = df
        self.base_config = base_config or Config()
        self.param_bounds = param_bounds or self.DEFAULT_PARAM_BOUNDS.copy()
        self.fixed_params = fixed_params or self.DEFAULT_FIXED_PARAMS.copy()
        self.logger = logger or logging.getLogger("SmartOptimizer")

        # 優化狀態
        self._study = None
        self._trials: List[TrialResult] = []
        self._best_value = float('-inf')
        self._convergence = []

    def _create_config(self, params: Dict) -> Config:
        """根據參數創建配置"""
        return Config(
            symbol=self.base_config.symbol,
            initial_balance=self.base_config.initial_balance,
            order_value=self.base_config.order_value,
            leverage=int(params.get('leverage', self.base_config.leverage)),
            take_profit_spacing=params.get('take_profit_spacing', self.base_config.take_profit_spacing),
            grid_spacing=params.get('grid_spacing', self.base_config.grid_spacing),
            direction=self.base_config.direction,
            max_drawdown=self.fixed_params.get('max_drawdown', 0.5),
            max_positions=self.fixed_params.get('max_positions', 50),
            fee_pct=self.fixed_params.get('fee_pct', 0.0004),
            position_threshold=self.fixed_params.get('position_threshold', 500.0),
            position_limit=self.fixed_params.get('position_limit', 100.0),
        )

    def _run_backtest(self, params: Dict) -> BacktestResult:
        """執行單次回測"""
        config = self._create_config(params)
        bt = GridBacktester(self.df.copy(), config)
        return bt.run()

    def _calculate_sortino_ratio(self, equity_curve: List[Tuple]) -> float:
        """計算 Sortino Ratio (只考慮下行風險)"""
        if len(equity_curve) < 2:
            return 0.0

        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1][2]
            curr_equity = equity_curve[i][2]
            if prev_equity > 0:
                returns.append((curr_equity - prev_equity) / prev_equity)

        if not returns:
            return 0.0

        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf') if avg_return > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf') if avg_return > 0 else 0.0

        # 年化 Sortino
        return (avg_return / downside_std) * np.sqrt(252)

    def _calculate_calmar_ratio(self, return_pct: float, max_drawdown: float) -> float:
        """計算 Calmar Ratio"""
        if max_drawdown == 0 or max_drawdown < 0.001:
            return float('inf') if return_pct > 0 else 0.0
        return return_pct / max_drawdown

    def _calculate_objective(
        self,
        result: BacktestResult,
        objective: OptimizationObjective
    ) -> float:
        """計算目標函數值"""

        if objective == OptimizationObjective.RETURN:
            return result.return_pct

        elif objective == OptimizationObjective.SHARPE:
            return result.sharpe_ratio

        elif objective == OptimizationObjective.SORTINO:
            return self._calculate_sortino_ratio(result.equity_curve)

        elif objective == OptimizationObjective.CALMAR:
            return self._calculate_calmar_ratio(result.return_pct, result.max_drawdown)

        elif objective == OptimizationObjective.PROFIT_FACTOR:
            return min(result.profit_factor, 10.0)  # 限制最大值

        elif objective == OptimizationObjective.RISK_ADJUSTED:
            # 風險調整收益: 收益 - 2*回撤
            return result.return_pct - 2 * result.max_drawdown

        else:
            return result.sharpe_ratio

    def _optuna_objective(
        self,
        trial: 'optuna.Trial',
        objective_type: OptimizationObjective
    ) -> float:
        """Optuna 目標函數"""
        start_time = time.time()

        # 從 Optuna 採樣參數
        params = {}

        # 獲取參數範圍
        tp_min, tp_max = self.param_bounds.get('take_profit_spacing', (0.001, 0.015))
        gs_min, gs_max = self.param_bounds.get('grid_spacing', (0.002, 0.025))

        # 限制 tp_max，確保 tp * 1.1 < gs_max (留空間給 grid_spacing)
        tp_max_safe = min(tp_max, gs_max / 1.15)
        if tp_max_safe < tp_min:
            tp_max_safe = tp_min * 2  # 至少有一些範圍

        # 止盈間距
        params['take_profit_spacing'] = trial.suggest_float(
            'take_profit_spacing', tp_min, tp_max_safe, log=True
        )

        # 補倉間距 (必須大於止盈間距)
        # 動態調整下限 (確保 gs > tp)
        gs_lower = max(gs_min, params['take_profit_spacing'] * 1.1)

        # 邊界檢查：如果下限超過上限，跳過此 trial
        if gs_lower >= gs_max:
            raise optuna.TrialPruned(f"Invalid param range: gs_lower={gs_lower:.4f} >= gs_max={gs_max:.4f}")

        params['grid_spacing'] = trial.suggest_float(
            'grid_spacing', gs_lower, gs_max, log=True
        )

        # 槓桿
        if 'leverage' in self.param_bounds:
            lev_min, lev_max = self.param_bounds['leverage']
            params['leverage'] = trial.suggest_int('leverage', int(lev_min), int(lev_max))
        else:
            params['leverage'] = self.base_config.leverage

        # 執行回測
        try:
            result = self._run_backtest(params)
            objective_value = self._calculate_objective(result, objective_type)

            # 處理無效值
            if np.isnan(objective_value) or np.isinf(objective_value):
                objective_value = -1e6

            # 記錄試驗
            duration = time.time() - start_time
            metrics = {
                'return_pct': result.return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'trades_count': result.trades_count,
                'win_rate': result.win_rate,
                'profit_factor': min(result.profit_factor, 100),
            }

            trial_result = TrialResult(
                trial_number=trial.number,
                params=params.copy(),
                metrics=metrics,
                objective_value=objective_value,
                duration=duration
            )
            self._trials.append(trial_result)

            # 更新收斂歷史
            if objective_value > self._best_value:
                self._best_value = objective_value
            self._convergence.append(self._best_value)

            return objective_value

        except Exception as e:
            self.logger.warning(f"Trial {trial.number} 失敗: {e}")
            return -1e6

    def _multi_objective(
        self,
        trial: 'optuna.Trial'
    ) -> Tuple[float, float, float]:
        """多目標優化函數 (最大化 Sharpe, 最小化回撤, 最大化收益)"""
        start_time = time.time()

        params = {}
        tp_min, tp_max = self.param_bounds.get('take_profit_spacing', (0.001, 0.015))
        params['take_profit_spacing'] = trial.suggest_float(
            'take_profit_spacing', tp_min, tp_max, log=True
        )

        gs_min, gs_max = self.param_bounds.get('grid_spacing', (0.002, 0.025))
        gs_lower = max(gs_min, params['take_profit_spacing'] * 1.2)
        params['grid_spacing'] = trial.suggest_float(
            'grid_spacing', gs_lower, gs_max, log=True
        )

        if 'leverage' in self.param_bounds:
            lev_min, lev_max = self.param_bounds['leverage']
            params['leverage'] = trial.suggest_int('leverage', int(lev_min), int(lev_max))

        try:
            result = self._run_backtest(params)

            duration = time.time() - start_time
            metrics = {
                'return_pct': result.return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'trades_count': result.trades_count,
                'win_rate': result.win_rate,
            }

            trial_result = TrialResult(
                trial_number=trial.number,
                params=params.copy(),
                metrics=metrics,
                objective_value=result.sharpe_ratio,  # 用 Sharpe 作為主要指標
                duration=duration
            )
            self._trials.append(trial_result)

            # 返回三個目標: (Sharpe, -回撤, 收益)
            # Optuna 默認最小化，所以 Sharpe 和收益要取負
            return (
                -result.sharpe_ratio,      # 最大化 Sharpe
                result.max_drawdown,       # 最小化回撤
                -result.return_pct         # 最大化收益
            )

        except Exception as e:
            self.logger.warning(f"Trial {trial.number} 失敗: {e}")
            return (1e6, 1e6, 1e6)

    def optimize(
        self,
        n_trials: int = 100,
        objective: OptimizationObjective = OptimizationObjective.SHARPE,
        method: OptimizationMethod = OptimizationMethod.TPE,
        n_startup_trials: int = 10,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        progress_callback: Callable[[int, int, float], None] = None,
        show_progress: bool = True
    ) -> SmartOptimizationResult:
        """
        執行智能優化

        Args:
            n_trials: 試驗次數
            objective: 優化目標
            method: 優化方法
            n_startup_trials: 隨機採樣次數 (用於 TPE 初始化)
            timeout: 超時時間 (秒)
            n_jobs: 並行數
            progress_callback: 進度回調 (current, total, best_value)
            show_progress: 是否顯示進度

        Returns:
            SmartOptimizationResult
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("請安裝 Optuna: pip install optuna")

        start_time = time.time()
        self._trials = []
        self._best_value = float('-inf')
        self._convergence = []

        self.logger.info(f"開始智能優化: 方法={method.value}, 目標={objective.value}, 試驗數={n_trials}")

        # 選擇採樣器
        if method == OptimizationMethod.TPE:
            sampler = TPESampler(
                n_startup_trials=n_startup_trials,
                multivariate=True,
                seed=42
            )
        elif method == OptimizationMethod.NSGA_II:
            sampler = NSGAIISampler(seed=42)
        elif method == OptimizationMethod.NSGA_III:
            sampler = NSGAIIISampler(seed=42)
        else:
            sampler = TPESampler(n_startup_trials=n_startup_trials, seed=42)

        # 剪枝器 (早期停止)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        # 多目標優化
        if objective == OptimizationObjective.MULTI_OBJECTIVE:
            study = optuna.create_study(
                directions=['minimize', 'minimize', 'minimize'],
                sampler=sampler if method in [OptimizationMethod.NSGA_II, OptimizationMethod.NSGA_III]
                        else NSGAIISampler(seed=42)
            )

            if show_progress:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            study.optimize(
                self._multi_objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress
            )

            # 提取 Pareto 前沿
            pareto_front = []
            for trial in study.best_trials:
                pareto_front.append({
                    'params': trial.params,
                    'sharpe': -trial.values[0],
                    'max_drawdown': trial.values[1],
                    'return_pct': -trial.values[2],
                })

            # 選擇最佳 (基於 Sharpe)
            best_trial = max(study.best_trials, key=lambda t: -t.values[0])
            best_params = best_trial.params
            best_metrics = {
                'sharpe_ratio': -best_trial.values[0],
                'max_drawdown': best_trial.values[1],
                'return_pct': -best_trial.values[2],
            }

        else:
            # 單目標優化
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )

            if show_progress:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            # 自定義回調
            def callback(study, trial):
                if progress_callback:
                    progress_callback(
                        trial.number + 1,
                        n_trials,
                        study.best_value if study.best_trial else 0
                    )

            study.optimize(
                lambda trial: self._optuna_objective(trial, objective),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress,
                callbacks=[callback] if progress_callback else None
            )

            pareto_front = None
            best_params = study.best_params
            best_trial_obj = self._trials[study.best_trial.number] if self._trials else None
            best_metrics = best_trial_obj.metrics if best_trial_obj else {}

        self._study = study

        # 計算參數重要性
        param_importance = {}
        try:
            importance = optuna.importance.get_param_importances(study)
            param_importance = dict(importance)
        except Exception:
            # 如果無法計算，使用基於方差的簡化版本
            param_importance = self._calculate_variance_importance()

        optimization_time = time.time() - start_time

        result = SmartOptimizationResult(
            best_params=best_params,
            best_metrics=best_metrics,
            best_objective=study.best_value if hasattr(study, 'best_value') else self._best_value,
            all_trials=self._trials,
            param_importance=param_importance,
            pareto_front=pareto_front,
            convergence_history=self._convergence,
            optimization_time=optimization_time,
            n_trials=len(self._trials),
            method=method.value,
            objective_type=objective.value
        )

        self.logger.info(f"優化完成: 耗時 {optimization_time:.1f}s, 最佳目標值={result.best_objective:.4f}")

        return result

    def _calculate_variance_importance(self) -> Dict[str, float]:
        """基於方差計算參數重要性 (備用方法)"""
        if not self._trials:
            return {}

        df = pd.DataFrame([t.to_dict() for t in self._trials])
        importance = {}

        for param in ['take_profit_spacing', 'grid_spacing', 'leverage']:
            col = f'param_{param}'
            if col in df.columns:
                # 計算參數值與目標值的相關性
                corr = abs(df[col].corr(df['objective']))
                importance[param] = corr if not np.isnan(corr) else 0.0

        # 正規化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return importance

    def quick_optimize(
        self,
        n_trials: int = 50,
        objective: str = "sharpe"
    ) -> SmartOptimizationResult:
        """
        快速優化 (便捷方法)

        Args:
            n_trials: 試驗次數
            objective: 優化目標 ("return", "sharpe", "sortino", "calmar", "risk_adjusted")

        Returns:
            SmartOptimizationResult
        """
        obj_map = {
            "return": OptimizationObjective.RETURN,
            "sharpe": OptimizationObjective.SHARPE,
            "sortino": OptimizationObjective.SORTINO,
            "calmar": OptimizationObjective.CALMAR,
            "profit_factor": OptimizationObjective.PROFIT_FACTOR,
            "risk_adjusted": OptimizationObjective.RISK_ADJUSTED,
            "multi": OptimizationObjective.MULTI_OBJECTIVE,
        }

        objective_enum = obj_map.get(objective.lower(), OptimizationObjective.SHARPE)

        return self.optimize(
            n_trials=n_trials,
            objective=objective_enum,
            method=OptimizationMethod.TPE,
            n_startup_trials=min(10, n_trials // 5),
            show_progress=True
        )

    def get_study(self) -> Optional['optuna.Study']:
        """獲取 Optuna Study 對象 (用於進階分析)"""
        return self._study

    def save_results(self, filepath: str, result: SmartOptimizationResult):
        """保存優化結果"""
        data = {
            'best_params': result.best_params,
            'best_metrics': result.best_metrics,
            'best_objective': result.best_objective,
            'param_importance': result.param_importance,
            'optimization_time': result.optimization_time,
            'n_trials': result.n_trials,
            'method': result.method,
            'objective_type': result.objective_type,
            'convergence_history': result.convergence_history,
            'trials': [t.to_dict() for t in result.all_trials]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"結果已保存至 {filepath}")


# === 便捷函數 ===

def smart_optimize_grid(
    df: pd.DataFrame,
    base_config: Config = None,
    n_trials: int = 100,
    objective: str = "sharpe",
    progress_callback: Callable = None
) -> SmartOptimizationResult:
    """
    智能網格優化便捷函數

    Args:
        df: K線數據
        base_config: 基礎配置
        n_trials: 試驗次數
        objective: 優化目標
        progress_callback: 進度回調

    Returns:
        SmartOptimizationResult
    """
    optimizer = SmartOptimizer(df, base_config)

    obj_map = {
        "return": OptimizationObjective.RETURN,
        "sharpe": OptimizationObjective.SHARPE,
        "sortino": OptimizationObjective.SORTINO,
        "calmar": OptimizationObjective.CALMAR,
        "risk_adjusted": OptimizationObjective.RISK_ADJUSTED,
    }

    return optimizer.optimize(
        n_trials=n_trials,
        objective=obj_map.get(objective.lower(), OptimizationObjective.SHARPE),
        method=OptimizationMethod.TPE,
        progress_callback=progress_callback
    )


# === 測試 ===

if __name__ == "__main__":
    # 測試智能優化器
    from .data_loader import DataLoader

    print("載入數據...")
    loader = DataLoader()
    df = loader.load("XRPUSDC", "2025-11-01", "2025-11-30")

    print("\n開始智能優化...")
    optimizer = SmartOptimizer(df)
    result = optimizer.quick_optimize(n_trials=30, objective="sharpe")

    print("\n" + str(result))
