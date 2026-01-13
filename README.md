# AS Grid Dragon 刷怪籠

**加密貨幣多交易所自動化網格交易系統**

[![Threads](https://img.shields.io/badge/Threads-@mr.__.l-000000?style=for-the-badge&logo=threads)](https://www.threads.com/@mr.__.l)

---

## 系統簡介

AS Grid Dragon（刷怪籠）是一套專業的加密貨幣網格交易機器人，採用 **Exchange Adapter Pattern** 架構，統一支援多家主流交易所。系統透過 WebSocket 即時監控市場價格，自動執行網格買賣策略。

### 支援交易所

| 交易所 | 合約類型 | 狀態 |
|--------|----------|------|
| Binance | USDT 永續 | ✅ 已支援 |
| Bybit | USDT 永續 | ✅ 已支援 |
| Bitget | USDT 永續 | ✅ 已支援 |
| Gate.io | USDT 永續 | ✅ 已支援 |

---

## 系統架構

```
┌─────────────────────────────────────────────────┐
│              GridBot 核心引擎                    │
│  • 網格策略邏輯     • 風險管理                   │
│  • Bandit 優化器    • 領先指標 (GLFT)           │
└─────────────────────┬───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │    ExchangeAdapter      │
         │      (抽象介面)          │
         └────────────┬────────────┘
                      │
    ┌─────────┬───────┼───────┬─────────┐
    │         │       │       │         │
┌───┴───┐ ┌───┴───┐ ┌─┴─┐ ┌───┴───┐
│Binance│ │ Bybit │ │Bit│ │Gate.io│
│Adapter│ │Adapter│ │get│ │Adapter│
└───────┘ └───────┘ └───┘ └───────┘
```

### 目錄結構

```
AS刷怪籠/
├── core/                    # 核心模組
│   ├── bot.py              # 網格機器人引擎
│   └── strategy.py         # 策略邏輯
│
├── exchanges/              # 交易所適配器
│   ├── base.py            # 抽象介面定義
│   ├── binance.py         # Binance 實作
│   ├── bybit.py           # Bybit 實作
│   ├── bitget.py          # Bitget 實作
│   └── gate.py            # Gate.io 實作
│
├── indicators/             # 指標系統
│   ├── bandit.py          # Multi-Armed Bandit 優化器
│   └── leading.py         # GLFT 領先指標
│
├── backtest/              # 回測模組
│   ├── backtester.py      # 回測引擎
│   └── data_loader.py     # 歷史數據載入
│
├── web/                   # Web 介面 (Streamlit)
│   └── pages/
│       ├── 1_📈_交易監控.py
│       ├── 2_⚙️_交易對管理.py
│       ├── 3_🔬_回測優化.py
│       └── 4_🛠️_設定.py
│
├── config/                # 配置檔
├── Dockerfile
└── docker-compose.yml
```

---

## Docker 部署

### 1. 安裝 Docker

下載並安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. 克隆專案

```bash
git clone https://github.com/thankcoom/as-grid-dragon.git
cd as-grid-dragon
```

### 3. 啟動服務

```bash
docker-compose up -d
```

### 4. 開啟介面

瀏覽器打開: **http://localhost:8501**

### 5. 查看日誌

```bash
docker logs -f as-grid-trading
```

### 6. 停止服務

```bash
docker-compose down
```

---

## Web 介面說明

### 📈 交易監控
- 即時價格與持倉狀態
- 網格訂單視覺化
- 盈虧統計圖表

### ⚙️ 交易對管理
- 新增/移除交易對
- 設定網格參數（數量、間距、投資額）
- 啟動/停止單一幣種

### 🔬 回測優化
- 歷史數據回測
- 參數優化搜尋
- 績效報告生成

### 🛠️ 設定
- 選擇交易所
- 輸入 API Key / Secret
- 槓桿與風控參數

---

## 配置說明

### 各交易所認證機制

| 交易所 | API Key | API Secret | Passphrase | 說明 |
|--------|---------|------------|------------|------|
| Binance | ✅ | ✅ | ❌ | 2 因素認證 |
| Bybit | ✅ | ✅ | ❌ | 2 因素認證 |
| Gate.io | ✅ | ✅ | ❌ | 2 因素認證 |
| **Bitget** | ✅ | ✅ | **✅ 必需** | **3 因素認證** |

> **為什麼 Bitget 需要 Passphrase？**
>
> 這是 Bitget 官方 API 的強制安全設計。創建 API Key 時必須設定一個 Passphrase（密碼短語），每次 API 請求都需要這三個憑證。即使 API Key 和 Secret 洩露，沒有 Passphrase 仍無法操作，提供額外一層安全保護。

### API 權限需求

| 權限 | 必要性 |
|------|--------|
| 讀取帳戶 | ✅ 必須 |
| 合約交易 | ✅ 必須 |
| 提現 | ❌ 不需要 |

### 網格參數

| 參數 | 說明 | 建議值 |
|------|------|--------|
| 網格數量 | 買賣單總數 | 10-20 |
| 網格間距 | 每格價差 % | 0.3-1.0% |
| 投資額 | 單幣種資金 | 依資金量 |
| 槓桿 | 合約槓桿倍數 | 5-20x |

---

## 核心功能

### 網格策略
在設定價格區間內，以固定間距自動掛單買低賣高

### Bandit 優化器
使用 Multi-Armed Bandit 演算法動態調整:
- 網格間距
- 倉位大小
- 止盈止損閾值

### GLFT 領先指標
Grid Leading Flow Trigger - 根據市場流動性調整網格密度

### 風險管理
- 單幣種倉位上限
- 總資金風險控制
- 自動止盈止損

---

## 風險提示

⚠️ **重要聲明**

- 加密貨幣交易具有高風險
- 過去績效不代表未來表現
- 請僅使用可承受損失的資金
- 建議先使用測試網 (Testnet) 驗證

---

## 作者

**Louis**

[![Threads](https://img.shields.io/badge/Threads-@mr.__.l-000000?style=flat-square&logo=threads)](https://www.threads.com/@mr.__.l)

---

## 交易所註冊

| 交易所 | 推薦連結 |
|--------|----------|
| Binance | [註冊](https://accounts.binance.com/register?ref=ASLOUIS) |
| Bybit | [註冊](https://www.bybit.com/invite?ref=B1MDMYE) |
| Bitget | [註冊](https://partner.bitget.fit/bg/aslouis) |
| Gate.io | [註冊](https://www.gatenode.xyz/signup/VLUSXFLFAQ?ref_type=103) |

---

## License

MIT License
