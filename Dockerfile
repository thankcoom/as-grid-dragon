# AS 網格交易系統 - Docker 映像
# 基於 Python 3.11 slim 版本

FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (用於編譯部分 Python 套件)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔案並安裝
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 複製應用程式碼
COPY . .

# 建立數據目錄
RUN mkdir -p /app/data

# 暴露 Streamlit 預設端口
EXPOSE 8501

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 啟動 Streamlit 應用
CMD ["streamlit", "run", "web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
