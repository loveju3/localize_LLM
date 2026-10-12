# 一起啟動／關閉網站、Neon 與 Modal

## 一次性設定

在專案 `.env` 補上以下設定；不要將金鑰提交 Git：

```dotenv
NEON_API_KEY=<Neon 管理 API key，不是資料庫密碼>
NEON_PROJECT_ID=<Neon project ID>
NEON_ENDPOINT_ID=<ep- 開頭的 compute endpoint ID>
MODAL_PROFILE=loveju3vup
MODAL_WORKSPACE=loveju3vup
MODAL_ENVIRONMENT=main
PROJECT_PORT=8000
PROJECT_START_TIMEOUT=900
```

上述 Modal profile/workspace 是本專案既有部署使用的名稱；其他帳號請填自己的設定並同步 `VLLM_BASE_URL`。此版管理 `deploy/modal_vllm.py` 中的 `localize-llm-qwen3` 推論部署，支援其標準 `https://<workspace>--localize-llm-qwen3-serve.modal.run/v1` 網址。訓練 App 與 RunPod 不在關閉範圍內。

既有 `DATABASE_URL`、`VLLM_BASE_URL`、`VLLM_API_KEY`、`APP_API_KEY`、工作區與 R2／embedding 設定也需正確。Neon 管理 API key 要能管理指定 compute。資料庫連線字串本身無法呼叫暫停 API。程式會核對 endpoint 的 host 與資料庫 host（支援 pooler），以及 Modal profile 所屬 workspace，避免操作錯誤資源。

安裝既有 Modal SDK 並完成 CLI 登入（若已完成可略過）：

```bash
.venv/bin/python -m pip install -r requirements-modal.txt
.venv/bin/python -m modal setup
```

Modal 的 `localize-llm-vllm` Secret、模型快取與 Neon 工作區需先依既有部署文件準備。啟動指令不建立 Secret，也不替使用者初始化資料庫或選擇共用模型。指定 profile 的憑證來自 Modal 本機設定；忽略 shell 中覆蓋 profile 的 token 變數。

## 日常操作

```bash
./project start
```

這是前景執行，保持終端機開著。流程：

1. 取得專案鎖並保留本機埠，防止重複啟動或衝突；不會殺掉占用埠的 Docker／其他程式。
2. 檢查雲端管理設定與資源對應。
3. 連線 Neon、核對工作區設定、取得預設模型。
4. 部署 Modal 推論程式（停止過的部署由此重建），請求 `/models` 喚醒 GPU，等待預設模型可用。
5. 只有上述檢查成功後才啟動網站，顯示 `/documents` 網址。

首次冷啟動可能需數分鐘。`PROJECT_START_TIMEOUT` 分別限制 Modal 部署與模型就緒等待。若網站或雲端啟動失敗，在已開始啟用資源的情況下會嘗試清理並回報結果。

在另一個終端機執行：

```bash
./project status
./project stop
```

也可在原本的前景終端機按 **Ctrl+C**，會執行相同清理流程。

關閉順序為：停止接受網站請求並等待進行中的請求結束（最多 20 秒）→ 停止 Modal 推論部署 → 暫停 Neon compute。即使 Modal 停止失敗，仍會嘗試暫停 Neon；任一服務未確認停止，指令回傳非零狀態，不會宣稱全部成功。

網站關閉是停止本機 HTTP 服務，不會關掉使用者的瀏覽器視窗或其他分頁。只管理由 `./project start` 啟動的網站；先前直接執行的 Uvicorn 或 Docker 需由原本方式關閉。若沒有受管理程序，`stop` 仍可依目前 `.env` 重試雲端清理，但不會終止其他本機程序。

本次停止操作不刪除 Neon 資料、Modal named Volumes 或 Secrets。Modal App 停止後必須重新部署，因此下次 `start` 會自動部署。這會影響共用同一 Neon compute／Modal 推論 App 的其他用戶端；請對專案專用資源使用統一關閉指令。

## 狀態與限制

- `status` 只讀本機鎖與 `data/runtime/state.json`，不呼叫雲端，因此不會喚醒 GPU 或資料庫。`ready` 表示啟動檢查通過且網站已開始服務，不是持續的雲端健康保證。
- 目前資料庫仍採每次操作開關連線；啟動檢查不建立永久資料庫連線池，也沒有持續 ping 保溫。Neon／Modal 仍可按雲端設定閒置縮減，下次请求再喚醒。
- Neon 暫停後仍可被其他用戶端的新資料庫連線喚醒；本機關閉無法禁止其他用戶端連線。
- 資料與模型儲存仍保留；停止運算不表示所有儲存費用歸零。
- 斷電、強制終止或網路中斷無法保證清理成功。恢復後執行 `./project stop` 重試，並檢查回報。若在執行期間改 `.env`，正常停止由原程序使用啟動時的設定；程序崩潰後重試則使用目前設定。
- `./project` 適用 macOS/Linux。直接 `uvicorn app.main:app` 保留原本按需連線行為，不會管理雲端停止。

## 離線驗證

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_lifecycle.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

測試使用模擬管理 API／推論服務，驗證資源身分、停止狀態確認、冷啟動重試、失敗清理、程序鎖與停止請求。依本次要求，沒有實際部署或停止遠端資源；Neon 管理 API 設定尚待補齊。

參考：[Neon suspend endpoint](https://api-docs.neon.tech/reference/suspendprojectendpoint)、[Modal deployment lifecycle](https://modal.com/docs/guide/managing-deployments)。
