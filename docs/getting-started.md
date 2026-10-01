# 第一版啟動與操作

## 1. 每台電腦建立環境

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

版本清單在 Apple Silicon macOS／Python 3.13 驗證；不同 OS 的套件 wheel 相容性需另測。本機不安裝 vLLM。

## 2. 設定共用雲端資源

在 `.env` 填入：

- `DATABASE_URL`：Neon PostgreSQL 連線字串，使用提供的 TLS 設定（例如 sslmode=require）。
- `WORKSPACE_ID`：所有要共用文件的機器使用相同值。
- `R2_ENDPOINT_URL`、`R2_BUCKET_NAME`、`R2_ACCESS_KEY_ID`、`R2_SECRET_ACCESS_KEY`：同一個私有 R2 bucket 的設定。
- `APP_API_KEY`：本機介面與 API 驗證用。可用 `python -c 'import secrets; print(secrets.token_urlsafe(32))'` 產生。
- `EMBEDDING_MODEL=BAAI/bge-m3`，以及它的固定 `EMBEDDING_REVISION`。

查詢一次模型 commit（僅查 metadata）：

```bash
python -m app.cli resolve-revision BAAI/bge-m3
```

將輸出的完整 SHA 填入 `.env`，各機器使用相同設定。不要在每台機器各自解析最新版本。`EMBEDDING_DEVICE=cpu` 是預設；Mac MPS 加速可另外測試後設為 `mps`。

首次初始化（需要可以建立 vector extension／資料表的資料庫權限）：

```bash
python -m app.cli init-db
```

此命令只建立缺少的初期資料表及工作區，不覆寫既有 embedding 設定。不是通用 schema migration 工具；後續資料表變更需額外 migration。

## 3. 啟動介面與建檔

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

開啟 http://127.0.0.1:8000，輸入 `APP_API_KEY`，連線後可上傳 PDF、查看共用文件、只搜尋文件。第一次 embedding 會下載模型到本機快取；尚未設定 GPU 仍可建檔及搜尋。

預設每份 PDF 上限 20MB、300 頁、3000 片段。頁碼是 PDF 物理頁序。掃描／低文字頁會顯示警告，完全無文字則拒絕建檔。問答可能不涵蓋未解析頁面。此版同步處理，上傳完成前請勿關閉程式。

`/health` 只代表本機程式運行，不代表雲端可用。`/api/workspace` 會檢查資料庫共用設定；文件上傳及模型列表才會測到相應雲端服務。

## 4. 登錄生成模型

```bash
python -m app.cli resolve-revision Qwen/Qwen3.5-4B
python -m app.cli register-base --id qwen35-4b-v1 \
  --base-model Qwen/Qwen3.5-4B --base-revision <完整模型SHA>
```

`<完整模型SHA>` 是要自行替換的佔位值。模型版本不可覆寫；變更權重請建立新 ID。

在 Linux GPU 主機準備相同專案及雲端設定，執行：

```bash
python -m app.cli prepare-model --id qwen35-4b-v1
```

命令會輸出 `vllm serve ...`，不會直接執行。GPU 主機需另行安裝／使用相容的 vLLM 容器。先在主機環境設定非空白 `VLLM_API_KEY`（vLLM 讀取此變數），再執行生成的啟動指令；同名模型必須維持相同權重。指令預設只監聽 localhost，以 SSH tunnel 或經驗證的 HTTPS proxy 供 Mac 存取。容器部署時，adapter 路徑必須掛載並在容器內可見。

在 Mac `.env` 填入 `VLLM_BASE_URL`（以 `/v1` 結尾）及相同 `VLLM_API_KEY`，重啟本機 API。重新整理模型列表，選擇 available 模型並設為共用預設，之後即可問答。

第一版沒有 GPU 自動啟停；到 Runpod 啟動與停止資源需自行完成，停止後檢查保留磁碟費用。

## 5. 上傳及選用 LoRA

準備包含 `adapter_config.json`、`adapter_model.safetensors` 的訓練輸出目錄：

```bash
python -m app.cli upload-lora --id report-lora-v1 \
  --directory ./models/training-output \
  --base-model Qwen/Qwen3-8B --base-revision <訓練時的基礎模型SHA>
```

此例只是示範 Qwen3-8B adapter，不是已訓練完成的模型。第一版接受標準固定 rank LoRA，拒絕 DoRA、額外 modules_to_save 及非 safetensors 權重。架構支援及效果仍需在實際 vLLM 驗證。

在 GPU 主機執行：

```bash
python -m app.cli prepare-model --id report-lora-v1 --output ./models
```

它會從 R2 下載 adapter、核對雜湊，並輸出含基礎模型 revision、LoRA rank 與 module 的指令。下載目錄若已存在會拒絕覆寫。等現有問答完成後再重啟服務，確認模型列表 available 才選用。

每次問答會保存模型快照、prompt 版本、引用原文與 token usage 到 Neon。第一版會保存問題與答案，請依內部資料保留需求使用。

## 6. API 與測試

API 文件位於 `/docs`。除首頁、健康檢查與 API 結構文件外，業務 API 均需 `Authorization: Bearer <APP_API_KEY>`。

```bash
python -m unittest discover -s tests -v
python -m pip check
```

本機 UI 使用文字節點顯示模型輸出及檔名。文件下載由 R2 短效簽署連結提供。此版本是受信任內部試用；對外部署前仍需個人登入、文件 ACL、請求配額與背景處理機制。
