# 真實雲端測試與更換 Pod

## 2026-10-02 驗證環境

- RunPod：使用者建立的 A40 48GB Pod `ziloubydlxhvsp`，畫面列價 US$0.50／小時。
- vLLM 0.30.0，服務模型名稱 `Qwen/Qwen3-8B`。
- 本次映像來源為 `vllm/vllm-openai:latest`；RunPod 日誌顯示 digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`。重建時應固定 digest，避免 latest 變更。
- GPU 主機模型快取只有一個 snapshot：`b968826d9c46dd6066d109eabc6255188de91218`；Neon 模型目錄以 `qwen3-8b-b968826d` 登錄。
- 目前 Pod 的啟動參數沒有 `--revision`，本次版本依實際快取記錄。正式重現部署時須加入 `--revision b968826d9c46dd6066d109eabc6255188de91218`；重新下載 main 後不可直接沿用舊版本登錄。
- 本機 BGE-M3 使用固定 revision `5617a9f61b028005a4858fdac845db406aefb181`，1024 維，CPU embedding。
- Neon `personal` 工作區已初始化；R2 已通過上傳、簽署下載與內容 SHA-256 一致性檢查。

## 已驗證功能

使用一頁合成 PDF，內容為 fictional project Cedar Lantern：2025 年營收 USD 1,200,000、營收成長 20%、員工 42 人，未提供 2035 年預測。

透過實際本機 HTTP API（不是 mock）驗證：

1. 工作區與模型清單，可用模型已設為共用預設。
2. 直接對話及第二輪記憶，兩輪均回答指定代號「青松四十二」。
3. PDF 建檔、重複上傳去重、文件清單、R2 原檔下載與 SHA-256 比對。
4. Neon 向量搜尋與文件範圍限制。
5. 中文文件問答回答營收成長 20%，引用正確文件第 1 頁。
6. 對未提供的 2035 年營收拒答。

第一輪測試約 61.9 秒；兩次對話約 4.9／4.5 秒，文件回答約 8.8 秒，拒答約 6.6 秒。這些是單次測量，不是效能保證或租用時間。請求耗時不包含全部 GPU 啟動、下載與閒置費用。

四題批次評估也已實際執行：完成 4 題、0 錯誤，拒答判斷 4/4、預期頁面檢索 3/3、引用 3/3、字串檢查 2/2。報告在 `data/cloud-validation/evaluation-run-1.jsonl`。瀏覽器另確認直接聊天與文件回答能顯示、引用有原文與頁碼、清除對話按鈕正常。

本機報告位於 `data/cloud-validation/live-run-1.jsonl`，合成文件與相關測試資料保留供重測，不包含私人文件。原始問題與文件問答結果也會依既有功能保存在 Neon；直接聊天不寫入問答紀錄。

## 重跑真實測試

先啟動 GPU／vLLM，確認模型與 `.env` 設定一致，再啟動本機：

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在另一個終端機執行：

```bash
.venv/bin/python -m app.cloud_check \
  --pdf data/cloud-validation/synthetic-report.pdf \
  --output data/cloud-validation/live-run-2.jsonl
```

這是會呼叫真實服務的選擇性測試，不包含在 unittest 中。需要此處描述的合成 PDF；不是任意文件的自動品質評估器。PDF 與結果放在忽略 Git 的 data 目錄，不隨 checkout 傳送。若要在別台機器重測，需複製合成 PDF 或建立含上述事實的一頁 PDF。每次使用新的輸出檔名；中途失敗回傳非零狀態並保留已完成步驟。

一般文件品質評估請使用 [批次評估工具](evaluation.md)。上述少量合成案例僅驗證串接，不代表真實中文報告、複雜表格或大量文件已達品質目標。

## 新建 Pod 後要更新什麼

1. `VLLM_BASE_URL`：新 Pod 的 HTTP 8000 服務網址，加上 `/v1`。舊 Pod 的 proxy URL 不會自動指向新 Pod。
2. `VLLM_API_KEY`：需與新 Pod 的設定相同。它與本機登入 UI 的 `APP_API_KEY` 不同。
3. 模型目錄：核對 `/v1/models`、實際基礎模型與 revision。換模型或權重版本要新增登錄；若只是換主機且模型完全相同，可沿用登錄。
4. 重啟本機 Uvicorn 以重新載入 `.env`，在網頁按「連線／重新整理」。

Neon、R2、APP_API_KEY、本機 embedding 快取與已索引文件通常都能沿用。Pod 內的套件與模型檔案能否沿用，取決於映像及是否掛載原本的持久儲存。不要因為換 Pod 就重建 embedding 索引。

GPU 顯示 Running 不代表模型 API 已就緒。檢查服務健康狀態、帶正確驗證的 `/v1/models` 及實際 completion。無 key 的 `/v1/models` 回應 401 代表驗證生效，不能當作服務壞掉。

本專案不自動啟停 GPU。測試完成應到 RunPod 停止 Pod，並另行確認 network volume 等持久儲存費用；停止 GPU 不代表所有儲存都免費。

本次這台 Pod 的停止確認頁顯示「You do not have a volume configured. ALL DATA will be lost!」。停止前必須確認 Pod 內資料可捨棄或完成備份；不能假設模型快取會保留。Neon／R2 與 Mac 本機資料不在該容器內。
