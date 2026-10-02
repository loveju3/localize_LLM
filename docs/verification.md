# 第一版驗證紀錄

## 2026-10-02：真實雲端串接與聊天 UI

- 36 項本機／模擬測試通過，6 項隔離 PostgreSQL 整合測試跳過。
- 另已實際呼叫 Neon、R2、RunPod Qwen3-8B，通過健康檢查、模型清單、多輪聊天、PDF 上傳／去重、原檔下載雜湊比對、文件範圍檢索、引用回答及無答案拒答。
- 四題真實批次評估完成，0 執行錯誤；拒答判斷 4/4、預期頁面檢索 3/3、引用 3/3、指定字串檢查 2/2。這是合成文件串接測試，不是一般品質評分。
- 實際瀏覽器驗證工作區連線、文件／模型清單、直接聊天、文件問答引用及清除聊天。使用獨立 localhost 測試實例及暫時 API key，未修改正式 APP_API_KEY。
- 詳細環境、實際模型版本限制及重測步驟見 [雲端驗證紀錄](cloud-validation.md)。
- LoRA 真實載入、GPU 自動啟停、20～30 題真實文件評估及成本基準尚未完成。

## 2026-10-02：批次評估工具

- 新增 12 項評估測試，31 項本機／模擬測試通過。
- 6 項 PostgreSQL 整合測試因未設定隔離測試資料庫而跳過，本次沒有重新驗證資料庫。
- 覆蓋多模型共用檢索片段、不修改預設模型、預期頁面命中率、拒答、文件範圍傳遞、未知模型、題庫格式、失敗續跑、錯誤資訊遮蔽、離線驗證與報告防覆寫。
- 沒有啟動 RunPod、下載模型或呼叫真實雲端服務；實際品質與成本仍待驗證。

## 2026-10-01：第一版

日期：2026-10-01。

- Python 3.13.7 / Apple Silicon macOS。
- 25 項測試通過：19 項本機／模擬服務測試、6 項真實 PostgreSQL／pgvector 整合測試。
- PostgreSQL 測試使用隔離的 `pgvector/pgvector:pg18` Docker 容器；測試後已停止並移除。
- 測試包括 PDF 抽取與切段、空白／掃描文件拒絕、重複上傳、引用 ID 驗證、無證據時跳過推論、API 驗證、LoRA 基礎模型檢查、safetensors 上傳下載與雜湊驗證、vLLM HTTP 請求內容。
- 真實 SQL 測試包含向量排序、文件範圍過濾、交易回滾、並行重複寫入、embedding 設定不一致拒絕、工作區隔離、模型選用與問答版本紀錄。
- `pip check` 通過，無依賴衝突。
- Uvicorn 實際啟動，HTTP `/health` 回應 200；首頁 HTML 與 API 路由由 TestClient 驗證。此環境無可用的瀏覽器控制介面，未完成瀏覽器視覺／互動驗證。

尚未驗證：真實 Neon／R2／vLLM 連線、下載實際 embedding 權重後的推論、雲端 GPU 顯存／速度／費用、真實 LoRA 在 vLLM 的載入、文件問答品質。

測試命令：

```bash
python -m unittest discover -s tests -v
# 另以 TEST_DATABASE_URL 指向可拋棄的 PostgreSQL + pgvector 資料庫，執行同一命令。
# 沒設定 TEST_DATABASE_URL 時，6 項資料庫測試會明確標記 skipped。
```

套件目前會輸出 Starlette TestClient 對 httpx 的棄用提醒；本次測試仍成功，尚未為此更換正式 HTTP client。
