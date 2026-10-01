# Localize LLM

以開源模型與雲端 vLLM 建立中英文文件問答。第一階段以少量 PDF、可靠回答與來源引用為主。

- [需求與架構討論整理](docs/project-plan.md)
- [雲端推論環境規劃](docs/cloud-runtime.md)
- [模型、算力與跨機器共用決策](docs/shared-cloud-design.md)
- [第一版啟動與操作](docs/getting-started.md)
- [驗證紀錄與未驗證項目](docs/verification.md)

## 本機環境

本機使用 Python 3.13 的 `.venv`，供文件處理、embedding 與呼叫雲端服務使用。

```bash
source .venv/bin/activate
python scripts/check_environment.py
python -m pip check
```

從乾淨 checkout 重建（鎖定依賴以本次 macOS／Python 3.13 環境為準）：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.in` 列出直接依賴；`requirements.txt` 記錄本次安裝版本。更新依賴後需重新檢查及產生鎖定清單：

```bash
python -m pip install -r requirements.in
python scripts/check_environment.py
python -m pip check
python -m pip freeze > requirements.txt
```

需要連接雲端服務時，複製 `.env.example` 為 `.env` 並填寫設定。`.env`、文件、模型與快取皆已排除 Git。應用程式會透過 Pydantic Settings 載入設定；範本不含真實金鑰。

## 啟動

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

開啟 http://127.0.0.1:8000 使用網頁介面，或 http://127.0.0.1:8000/docs 使用 API 文件。無雲端設定也能開啟介面與 `/health`；實際建檔、搜尋及問答需要完成 [操作說明](docs/getting-started.md) 的設定。

已實作 PDF 建檔、R2 原檔儲存、Neon 向量查詢、vLLM 問答、來源檢視、模型目錄與共用預設切換、LoRA 上傳／下載校驗。資料庫初始化、模型登錄及 adapter 準備透過 `python -m app.cli --help` 操作。

初期是一個受信任內部工作區，所有持有工作區 API key 的人可查看該工作區資料及切換預設模型；不是對外多租戶權限系統。每台電腦自行執行本機介面並連接相同 Neon／R2。

## 驗證

```bash
python -m unittest discover -s tests -v
python -m pip check
```

單元測試使用模擬雲端服務，不會下載模型或啟動 GPU。真實雲端連線與模型品質需在設定服務後驗證。OCR、複雜表格解析、混合檢索、GPU 自動啟停與 LoRA 訓練流程尚未實作。
