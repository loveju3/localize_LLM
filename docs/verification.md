# 第一版驗證紀錄

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
