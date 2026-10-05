# RunPod／Modal 相容部署

本機 UI、Neon、R2、embedding 與模型登錄保持共用。兩個供應商都提供帶 Bearer key 的 OpenAI-compatible `/v1/models` 與 `/v1/chat/completions`；`VLLM_BASE_URL` 是介面名稱，不限定 RunPod。

## Modal 設定

- Workspace：`loveju3vup`，environment：`main`，App：`localize-llm-qwen3`。
- Qwen3-8B 固定 revision、vLLM 固定映像 digest，見 `deploy/modal_vllm.py`。
- GPU：L40S，備選 A100-40GB；最多 1 容器、最少 0，閒置 120 秒後可縮到零。啟動、保溫和執行都會用到計費資源；不保證排程立即取得 GPU。
- CPU 4 核、RAM 16GiB。啟動上限 900 秒，單次請求執行上限 600 秒。
- Volume：`localize-llm-model-cache`，模型先用 CPU 任務下載、驗證 config.json 並 commit，再啟動 GPU 推論。
- Secret：`localize-llm-vllm`，只包含專用 `VLLM_API_KEY`。不傳送 Neon／R2／APP_API_KEY 到 GPU 容器。
- 目前只部署基礎模型推論，沒有新增 LoRA 訓練或 adapter 載入功能。

## 重現部署

```bash
.venv/bin/python -m pip install -r requirements-modal.txt
# 先登入 Modal CLI，確認 active profile 和工作區。
.venv/bin/python -m modal app list --env main --profile loveju3vup
# .env.modal 只保存這個端點的金鑰與網址，檔案已被 Git 排除。
.venv/bin/python -m modal secret create localize-llm-vllm --from-dotenv .env.modal --env main --profile loveju3vup
.venv/bin/python -m modal deploy deploy/modal_vllm.py --env main --profile loveju3vup
.venv/bin/python -m modal run --env main --profile loveju3vup deploy/modal_vllm.py
```

建立 Secret 不使用 `--force`，避免意外覆寫。將部署輸出的實際 HTTPS 網址加 `/v1`，保存到 `.env.modal` 的 `VLLM_BASE_URL`。不要把金鑰貼到 Git 或日誌。

## 切換供應商

首次切換前，保存現有 RunPod 設定：

```bash
.venv/bin/python deploy/select_backend.py runpod --save-current
.venv/bin/python deploy/select_backend.py modal
# 重啟本機 Uvicorn，再在 UI 按「連線／重新整理」。
```

切回：

```bash
.venv/bin/python deploy/select_backend.py runpod
```

工具只替換 `.env` 的端點、推論金鑰與等待時間，保留資料庫、R2 與工作區登入。Modal 等待時間 600 秒，RunPod 180 秒；第一次 Modal 呼叫可能因模型載入而較慢。切換前需確認目的端點已部署且模型版本相同。切換不會關閉另一個供應商的機器，也不會自動故障轉移。

## 驗證

先確認無金鑰 `/v1/models` 為 401，正確金鑰可列出 Qwen3-8B，再切換並重新啟動本機。

```bash
.venv/bin/python -m app.cloud_check --pdf data/cloud-validation/synthetic-report.pdf --output data/cloud-validation/modal-live-1.jsonl
.venv/bin/python -m app.evaluation data/cloud-validation/questions.jsonl --model qwen3-8b-b968826d --output data/cloud-validation/modal-evaluation-1.jsonl
```

合成 PDF／題庫是本機忽略檔案，重測方法見 `cloud-validation.md`。冷啟動、排程與正式文件品質須另外測量，單次合成案例不是 SLA 或品質保證。

參考：[Modal web server](https://modal.com/docs/reference/modal.web_server)、[Secrets](https://modal.com/docs/guide/secrets)、[Volumes](https://modal.com/docs/guide/volumes)。

## 2026-10-05 實際部署與驗證

- 已部署到 [loveju3vup／main](https://modal.com/apps/loveju3vup/main/deployed/localize-llm-qwen3)，實際服務為 `https://loveju3vup--localize-llm-qwen3-serve.modal.run/v1`。
- 原映像需要 `add_python="3.12"` 供 Modal runtime 使用；vLLM 仍由映像內的 executable 執行。CPU 下載任務使用獨立 Python 3.12／huggingface-hub 0.36.0 映像。
- 固定 revision 模型已下載、config.json 驗證並 commit 至 Volume。
- 端點驗證：不帶金鑰 `/v1/models` 為 401，帶專用金鑰為 200；實際對話回答「你好！」。首次探測含冷啟動共 109.52 秒，不是純生成時間或效能保證。
- 本機 `.env` 已切到 Modal，Uvicorn 已重新啟動；RunPod 端點備份保存在忽略的 `.env.runpod`，可用上述工具切回。
- `app.cloud_check` 所有步驟通過：工作區、模型、兩輪對話、PDF 上傳／去重／清單、R2 下載雜湊、Neon 搜尋、帶引用問答、無答案拒答。報告：`data/modal-validation/live-1.jsonl`。
- 四題評估：4 completed、0 errors，拒答判斷 4/4、檢索頁碼 3/3、引用頁碼 3/3、字串檢查 2/2。報告：`data/modal-validation/evaluation-1.jsonl`。
- 網頁實測：模型清單 available，對話回答「是的，我現在可以正常回應。」；該次請求 4.92 秒。
- 42 項單元測試通過，其中 6 項需拋棄式 PostgreSQL 的測試略過；`pip check` 與 `git diff --check` 通過。

目前驗證涵蓋基礎模型推論與合成 PDF 的真實串接；2026-10-05 已另完成 GPU LoRA 訓練與頁面上傳／載入 API 實測，見 `lora-training.md` 與 `lora-upload.md`。尚未驗證大量併發或正式文件品質。部署設定為閒置縮到零；切換供應商不會自動停止 RunPod，需另行管理其費用。
