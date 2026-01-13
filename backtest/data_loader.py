"""
數據載入模組
支援多種交易對的歷史數據下載與載入
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Union
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 嘗試導入統一路徑解析器（2025 最佳實踐）
try:
    # 需要先將項目根目錄加入 sys.path
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from core.path_resolver import get_backtest_data_dir, get_asback_path
    _PATH_RESOLVER_AVAILABLE = True
except ImportError:
    _PATH_RESOLVER_AVAILABLE = False
    logger.debug("core.path_resolver 不可用，使用備用路徑解析")


class DataLoader:
    """數據載入器 - 支援多交易對"""

    # 支援的交易對列表
    SUPPORTED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "XRPUSDC", "XRPUSDT",
        "SOLUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT"
    ]

    def __init__(self, data_dir: str = None):
        """
        初始化數據載入器

        Args:
            data_dir: 數據目錄路徑，預設為 ./data
        """
        if data_dir is None:
            self.data_dir = self._get_default_data_dir()
        else:
            self.data_dir = Path(data_dir)

    def _get_default_data_dir(self) -> Path:
        """
        獲取預設數據目錄（支援打包後的環境）
        使用 2025 最佳實踐：優先使用統一路徑解析器
        """
        # 方法 1：使用統一路徑解析器（最可靠）
        if _PATH_RESOLVER_AVAILABLE:
            try:
                data_dir = get_backtest_data_dir()
                if data_dir and data_dir.exists():
                    logger.debug(f"使用統一路徑解析器: {data_dir}")
                    return data_dir
            except Exception as e:
                logger.debug(f"統一路徑解析器失敗: {e}")

        # 方法 2：Nuitka 編譯環境 - 使用 __compiled__ API
        if "__compiled__" in dir():
            try:
                compiled = __compiled__  # noqa: F821
                if hasattr(compiled, 'containing_dir'):
                    app_dir = Path(compiled.containing_dir)
                    data_dir = app_dir / "asBack" / "data"
                    if data_dir.exists():
                        logger.debug(f"Nuitka __compiled__: {data_dir}")
                        return data_dir
            except Exception:
                pass

        # 方法 3：備用手動搜索
        possible_paths = [
            # 開發環境：backtest_system 的上層目錄
            Path(__file__).parent.parent / "data",
            # Nuitka 打包後：與執行檔同目錄
            Path(sys.executable).parent / "asBack" / "data",
            # macOS .app bundle 結構
            Path(sys.executable).parent.parent / "Resources" / "asBack" / "data",
            # macOS .app bundle 內（直接）
            Path(sys.executable).parent / "data",
            # 當前工作目錄
            Path.cwd() / "asBack" / "data",
        ]

        for path in possible_paths:
            if path.exists():
                logger.debug(f"備用路徑搜索找到: {path}")
                return path.resolve()

        # 如果都不存在，使用用戶家目錄下的快取並建立
        cache_dir = Path.home() / ".as_grid" / "backtest_data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"使用用戶快取目錄: {cache_dir}")
        return cache_dir

    def get_data_path(self, symbol: str, date_str: str, interval: str = "1m") -> Path:
        """
        取得數據文件路徑

        Args:
            symbol: 交易對符號
            date_str: 日期字串 (YYYY-MM-DD)
            interval: K線間隔 (1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            Path: 數據文件路徑
        """
        return self.data_dir / f"futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.csv"

    def load_single_day(self, symbol: str, date_str: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        載入單日數據

        Args:
            symbol: 交易對符號
            date_str: 日期字串 (YYYY-MM-DD)
            interval: K線間隔

        Returns:
            DataFrame 或 None
        """
        path = self.get_data_path(symbol, date_str, interval)

        if not path.exists():
            return None

        try:
            df = pd.read_csv(path)
            if df.empty:
                return None

            # 標準化時間欄位
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            elif 'timestamp' in df.columns:
                df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 確保有必要的欄位
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"警告: {path} 缺少欄位 {col}")
                    return None

            return df

        except Exception as e:
            print(f"載入失敗 {path}: {e}")
            return None

    def load(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        載入日期範圍內的數據

        Args:
            symbol: 交易對符號
            start_date: 開始日期 (str: YYYY-MM-DD 或 datetime)
            end_date: 結束日期 (str: YYYY-MM-DD 或 datetime)
            interval: K線間隔

        Returns:
            合併後的 DataFrame
        """
        # 轉換日期格式
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = start_date

        if isinstance(end_date, str):
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = end_date

        # 生成所有日期列表
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        # 並行載入所有日期的數據 (使用多線程加速 I/O)
        all_data = []
        n_workers = min(8, len(date_list))  # 最多 8 個線程

        if n_workers > 1 and len(date_list) > 3:
            # 並行載入
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.load_single_day, symbol, date_str, interval): date_str
                    for date_str in date_list
                }
                for future in as_completed(futures):
                    df = future.result()
                    if df is not None:
                        all_data.append(df)
        else:
            # 少量數據用順序載入
            for date_str in date_list:
                df = self.load_single_day(symbol, date_str, interval)
                if df is not None:
                    all_data.append(df)

        if not all_data:
            raise ValueError(f"找不到 {symbol} 從 {start_date} 到 {end_date} 的數據")

        # 合併並排序
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.sort_values('open_time').reset_index(drop=True)

        print(f"✅ 載入 {symbol} 數據: {len(full_df):,} 條 K 線")
        print(f"   時間範圍: {full_df['open_time'].min()} 至 {full_df['open_time'].max()}")

        return full_df

    def load_symbol_data(
        self,
        symbol: str,
        timeframe: str = "1m",
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        便捷方法：載入最近 N 天的數據

        Args:
            symbol: 交易對符號 (如 "XRPUSDC", "BTCUSDT")
            timeframe: K線間隔 (如 "1m", "5m", "1h")
            days: 載入天數

        Returns:
            DataFrame 或 None
        """
        # 標準化 symbol (移除斜線等)
        symbol = symbol.upper().replace("/", "").replace(":", "").split(":")[0]

        # 計算日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            df = self.load(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe
            )
            return df
        except ValueError as e:
            print(f"載入數據失敗: {e}")
            return None
        except Exception as e:
            print(f"載入數據時發生錯誤: {e}")
            return None

    def list_available_data(self, symbol: str = None, interval: str = "1m") -> List[dict]:
        """
        列出可用的數據文件

        Args:
            symbol: 交易對符號 (None 表示所有)
            interval: K線間隔

        Returns:
            可用數據列表
        """
        available = []
        base_path = self.data_dir / "futures/um/daily/klines"

        if not base_path.exists():
            return available

        symbols_to_check = [symbol] if symbol else self.SUPPORTED_SYMBOLS

        for sym in symbols_to_check:
            sym_path = base_path / sym / interval

            if not sym_path.exists():
                continue

            for file in sorted(sym_path.glob(f"{sym}-{interval}-*.csv")):
                # 從文件名提取日期
                date_str = file.stem.split('-')[-1]
                # 嘗試解析完整日期
                try:
                    parts = file.stem.replace(f"{sym}-{interval}-", "").split("-")
                    if len(parts) == 3:
                        date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
                except:
                    pass

                available.append({
                    "symbol": sym,
                    "date": date_str,
                    "interval": interval,
                    "path": str(file)
                })

        return available

    def get_date_range(self, symbol: str, interval: str = "1m") -> Optional[tuple]:
        """
        取得某交易對可用的日期範圍

        Returns:
            (start_date, end_date) 或 None
        """
        available = self.list_available_data(symbol, interval)

        if not available:
            return None

        dates = [d["date"] for d in available]
        return min(dates), max(dates)

    def download(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1m"
    ) -> bool:
        """
        下載歷史數據 (使用 ccxt)

        Args:
            symbol: 交易對符號
            start_date: 開始日期
            end_date: 結束日期
            interval: K線間隔

        Returns:
            是否成功
        """
        try:
            import ccxt
        except ImportError:
            print("❌ 請先安裝 ccxt: pip install ccxt")
            return False

        # 轉換日期格式
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = start_date

        if isinstance(end_date, str):
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = end_date

        # 初始化交易所
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # 時間框架對應
        timeframe_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        timeframe = timeframe_map.get(interval, "1m")

        print(f"📥 下載 {symbol} 數據...")
        print(f"   時間範圍: {start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}")

        current = start
        total_bars = 0

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            output_path = self.get_data_path(symbol, date_str, interval)

            # 如果文件已存在，跳過
            if output_path.exists():
                current += timedelta(days=1)
                continue

            # 建立目錄
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 下載當天數據
            since = int(datetime(current.year, current.month, current.day).timestamp() * 1000)
            until = since + 24 * 60 * 60 * 1000  # 24小時

            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol.replace("USDC", "/USDC").replace("USDT", "/USDT"),
                    timeframe,
                    since=since,
                    limit=1500
                )

                if ohlcv:
                    # 只保留當天數據
                    ohlcv = [bar for bar in ohlcv if bar[0] < until]

                    df = pd.DataFrame(ohlcv, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume'
                    ])
                    df.to_csv(output_path, index=False)
                    total_bars += len(df)
                    print(f"   ✅ {date_str}: {len(df)} 條")

            except Exception as e:
                print(f"   ❌ {date_str}: {e}")

            current += timedelta(days=1)

        print(f"\n✅ 下載完成，共 {total_bars:,} 條數據")
        return True


# 使用範例
if __name__ == "__main__":
    loader = DataLoader()

    # 列出可用數據
    print("可用數據:")
    for item in loader.list_available_data("XRPUSDC")[:5]:
        print(f"  {item['symbol']} - {item['date']}")

    # 載入數據
    # df = loader.load("XRPUSDC", "2025-10-27", "2025-11-25")
    # print(df.head())
