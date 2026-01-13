"""
參數優化器模組
網格搜尋與參數優化
"""
import pandas as pd
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from .config import Config
from .backtester import GridBacktester, BacktestResult


@dataclass
class OptimizationResult:
    """優化結果"""
    best_config: Config
    best_result: BacktestResult
    all_results: pd.DataFrame
    param_importance: Dict[str, float]

    def __str__(self) -> str:
        return (
            f"優化結果\n"
            f"{'='*50}\n"
            f"最佳參數:\n"
            f"  止盈間距: {self.best_config.take_profit_spacing*100:.2f}%\n"
            f"  補倉間距: {self.best_config.grid_spacing*100:.2f}%\n"
            f"  槓桿: {self.best_config.leverage}x\n"
            f"\n最佳績效:\n"
            f"  收益率: {self.best_result.return_pct*100:.2f}%\n"
            f"  最大回撤: {self.best_result.max_drawdown*100:.2f}%\n"
            f"  交易次數: {self.best_result.trades_count}\n"
            f"  勝率: {self.best_result.win_rate*100:.1f}%\n"
            f"\n測試組數: {len(self.all_results)}"
        )


class GridOptimizer:
    """網格參數優化器"""

    # 預設參數範圍 (擴展版)
    DEFAULT_PARAM_RANGES = {
        # 止盈間距: 0.1% ~ 1.0% (更細粒度)
        "take_profit_spacing": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01],
        # 補倉間距: 0.2% ~ 2.0% (更寬範圍)
        "grid_spacing": [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02],
        # 槓桿
        "leverage": [5, 10, 15, 20, 25, 30]
    }

    def __init__(
        self,
        df: pd.DataFrame,
        base_config: Config = None,
        param_ranges: Dict[str, List] = None
    ):
        """
        初始化優化器

        Args:
            df: K線數據
            base_config: 基礎配置 (其他參數固定)
            param_ranges: 參數範圍 (key: 參數名, value: 參數值列表)
        """
        self.df = df
        self.base_config = base_config or Config()
        self.param_ranges = param_ranges or self.DEFAULT_PARAM_RANGES
        self.results: List[Dict] = []

    def _create_config(self, params: Dict) -> Config:
        """根據參數創建配置"""
        config_dict = self.base_config.to_dict()

        # 更新參數
        for key, value in params.items():
            if key in config_dict:
                config_dict[key] = value

        # 重新計算 long/short settings
        return Config.from_dict({
            k: v for k, v in config_dict.items()
            if k not in ['long_settings', 'short_settings']
        })

    def _run_single_backtest(self, params: Dict) -> Dict:
        """執行單次回測"""
        config = self._create_config(params)
        bt = GridBacktester(self.df.copy(), config)
        result = bt.run()

        return {
            **params,
            "return_pct": result.return_pct,
            "max_drawdown": result.max_drawdown,
            "trades": result.trades_count,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "final_equity": result.final_equity,
            "realized_pnl": result.realized_pnl,
            "unrealized_pnl": result.unrealized_pnl
        }

    def generate_param_combinations(self) -> List[Dict]:
        """生成所有參數組合"""
        keys = list(self.param_ranges.keys())
        values = list(self.param_ranges.values())

        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))

            # 過濾無效組合 (止盈間距應小於補倉間距)
            if "take_profit_spacing" in param_dict and "grid_spacing" in param_dict:
                if param_dict["take_profit_spacing"] >= param_dict["grid_spacing"]:
                    continue

            combinations.append(param_dict)

        return combinations

    def run(
        self,
        metric: str = "return_pct",
        ascending: bool = False,
        n_jobs: int = 1,
        progress_callback: Callable[[int, int], None] = None
    ) -> OptimizationResult:
        """
        執行網格搜尋優化

        Args:
            metric: 優化目標指標 (return_pct, sharpe_ratio, profit_factor 等)
            ascending: 是否升序排列 (False = 取最大值)
            n_jobs: 並行數量 (1 = 單線程)
            progress_callback: 進度回調函數 (current, total)

        Returns:
            OptimizationResult: 優化結果
        """
        combinations = self.generate_param_combinations()
        total = len(combinations)

        print(f"🔍 開始網格搜尋優化")
        print(f"   參數組合數: {total}")
        print(f"   優化目標: {metric}")
        print(f"   並行數量: {n_jobs}")
        print("="*50)

        self.results = []

        if n_jobs == 1:
            # 單線程執行
            for i, params in enumerate(combinations):
                result = self._run_single_backtest(params)
                self.results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)
                else:
                    self._print_progress(i + 1, total, result)
        else:
            # 多進程執行 (注意: 需要在 if __name__ == "__main__" 中使用)
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(self._run_single_backtest, params): params
                    for params in combinations
                }

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)

                    if progress_callback:
                        progress_callback(i + 1, total)

        # 轉換為 DataFrame 並排序
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(metric, ascending=ascending)

        # 取得最佳結果
        best_row = df_results.iloc[0]
        best_params = {k: best_row[k] for k in self.param_ranges.keys()}
        best_config = self._create_config(best_params)

        # 重新執行最佳配置以取得完整結果
        bt = GridBacktester(self.df.copy(), best_config)
        best_result = bt.run()

        # 計算參數重要性 (簡化版: 基於方差)
        param_importance = self._calculate_param_importance(df_results, metric)

        print("\n" + "="*50)
        print("✅ 優化完成")

        return OptimizationResult(
            best_config=best_config,
            best_result=best_result,
            all_results=df_results,
            param_importance=param_importance
        )

    def _print_progress(self, current: int, total: int, result: Dict):
        """打印進度"""
        params_str = ", ".join([
            f"{k}={v*100:.1f}%" if isinstance(v, float) and v < 1 else f"{k}={v}"
            for k, v in result.items()
            if k in self.param_ranges
        ])
        print(f"[{current}/{total}] {params_str} -> "
              f"收益: {result['return_pct']*100:.2f}%, "
              f"回撤: {result['max_drawdown']*100:.2f}%")

    def _calculate_param_importance(self, df: pd.DataFrame, metric: str) -> Dict[str, float]:
        """計算參數重要性"""
        importance = {}

        for param in self.param_ranges.keys():
            if param not in df.columns:
                continue

            # 計算每個參數值對應的平均指標值的方差
            grouped = df.groupby(param)[metric].mean()
            importance[param] = grouped.std() if len(grouped) > 1 else 0

        # 正規化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return importance

    def run_symmetric_search(
        self,
        spacings: List[float] = None
    ) -> pd.DataFrame:
        """
        對稱間距搜尋 (止盈=補倉)

        Args:
            spacings: 間距列表

        Returns:
            結果 DataFrame
        """
        if spacings is None:
            spacings = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02]

        results = []
        for spacing in spacings:
            config = Config(
                symbol=self.base_config.symbol,
                initial_balance=self.base_config.initial_balance,
                order_value=self.base_config.order_value,
                leverage=self.base_config.leverage,
                take_profit_spacing=spacing,
                grid_spacing=spacing,
                max_drawdown=self.base_config.max_drawdown,
                max_positions=self.base_config.max_positions,
                fee_pct=self.base_config.fee_pct,
                direction=self.base_config.direction
            )

            bt = GridBacktester(self.df.copy(), config)
            result = bt.run()

            results.append({
                "spacing": spacing,
                "spacing_pct": f"{spacing*100:.1f}%",
                "return_pct": result.return_pct,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades_count,
                "win_rate": result.win_rate
            })

            print(f"對稱間距 {spacing*100:.1f}%: "
                  f"收益 {result.return_pct*100:.2f}%, "
                  f"回撤 {result.max_drawdown*100:.2f}%")

        return pd.DataFrame(results)

    def run_asymmetric_search(
        self,
        take_profits: List[float] = None,
        grid_spacings: List[float] = None
    ) -> pd.DataFrame:
        """
        非對稱間距搜尋

        Args:
            take_profits: 止盈間距列表
            grid_spacings: 補倉間距列表

        Returns:
            結果 DataFrame
        """
        if take_profits is None:
            take_profits = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]
        if grid_spacings is None:
            grid_spacings = [0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02]

        results = []
        for tp in take_profits:
            for gs in grid_spacings:
                if tp >= gs:  # 止盈應小於補倉
                    continue

                config = Config(
                    symbol=self.base_config.symbol,
                    initial_balance=self.base_config.initial_balance,
                    order_value=self.base_config.order_value,
                    leverage=self.base_config.leverage,
                    take_profit_spacing=tp,
                    grid_spacing=gs,
                    max_drawdown=self.base_config.max_drawdown,
                    max_positions=self.base_config.max_positions,
                    fee_pct=self.base_config.fee_pct,
                    direction=self.base_config.direction
                )

                bt = GridBacktester(self.df.copy(), config)
                result = bt.run()

                results.append({
                    "take_profit": tp,
                    "grid_spacing": gs,
                    "tp_pct": f"{tp*100:.1f}%",
                    "gs_pct": f"{gs*100:.1f}%",
                    "return_pct": result.return_pct,
                    "max_drawdown": result.max_drawdown,
                    "trades": result.trades_count,
                    "win_rate": result.win_rate,
                    "sharpe_ratio": result.sharpe_ratio
                })

                print(f"止盈 {tp*100:.1f}% / 補倉 {gs*100:.1f}%: "
                      f"收益 {result.return_pct*100:.2f}%, "
                      f"回撤 {result.max_drawdown*100:.2f}%")

        return pd.DataFrame(results).sort_values("return_pct", ascending=False)

    def compare_directions(self) -> pd.DataFrame:
        """
        比較不同方向策略

        Returns:
            結果 DataFrame
        """
        directions = ["long", "short", "both"]
        results = []

        for direction in directions:
            config = Config(
                symbol=self.base_config.symbol,
                initial_balance=self.base_config.initial_balance,
                order_value=self.base_config.order_value,
                leverage=self.base_config.leverage,
                take_profit_spacing=self.base_config.take_profit_spacing,
                grid_spacing=self.base_config.grid_spacing,
                max_drawdown=self.base_config.max_drawdown,
                max_positions=self.base_config.max_positions,
                fee_pct=self.base_config.fee_pct,
                direction=direction
            )

            bt = GridBacktester(self.df.copy(), config)
            result = bt.run()

            results.append({
                "direction": direction,
                "return_pct": result.return_pct,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades_count,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio
            })

            print(f"方向 {direction}: "
                  f"收益 {result.return_pct*100:.2f}%, "
                  f"回撤 {result.max_drawdown*100:.2f}%")

        return pd.DataFrame(results)
