# 模型、算力與跨機器共用決策

整理日期：2026-10-01。補充先前討論；推薦項目仍需用實際文件與 GPU 驗證。

2026-10-02 更新：使用者另建的 RunPod A40／Qwen3-8B 已完成真實串接測試；本文件的候選與早期「未部署」描述是原始規劃。實際版本、費率及驗證界線以 [雲端驗證紀錄](cloud-validation.md) 為準。

## 模型候選

| 功能 | 第一個候選 | 比較候選 |
| --- | --- | --- |
| 生成答案 | Qwen3.5-4B | Qwen3.5-9B、Qwen3-8B、Gemma 3 4B IT |
| 產生向量 | BAAI/bge-m3，Mac 執行 dense embedding | Qwen3-Embedding-0.6B |

上述生成模型均有 vLLM 支援，但仍須固定並測試框架、模型 revision、GPU 與量化方式的組合。支援模型架構不代表所有 LoRA／量化格式都可用。第一版不是模型品質排名；先固定檢索結果比較 4B 與 9B。

第一版不自動下載任何生成模型，也不自行選取可變的 main revision。模型與 embedding 均記錄完整 commit SHA。4B BF16 權重約 8GB 只是參數概算，不能當總顯存；還需考慮推論快取、視覺元件與並行請求。先以 24GB GPU、單請求、8K 上下文測試。

## 算力推薦與預算

初期推薦 Runpod Pods，先測 RTX A5000 24GB。2026-10-01 查閱公開頁面列價為 US$0.27／小時；L4 24GB US$0.49、RTX 4090 24GB US$0.74、RTX A6000 48GB US$0.53。供應、區域及實際價格以建立資源時為準，不保證有貨。

若僅為預算估算假設 US$1=NT$33，NT$450 約可支付 A5000 50 小時運行，其餘 NT$150 留作儲存及緩衝；此非即時匯率與報價。啟動、載入、閒置皆可能計費，保留磁碟可能在停機後繼續計費。

本次沒有建立 Runpod 資源。自動啟停尚未實作，現階段需手動啟動及確認終止；執行 API 並不代表 GPU 會自動開機。

## 跨機器共用

- Neon 儲存工作區設定、文件清單、原文片段、向量、模型目錄、共用預設模型及問答紀錄。
- R2 儲存 PDF 與 LoRA adapter。原始文件以工作區及內容 SHA-256 定位，避免同內容重複建檔。
- 每台電腦使用相同資料庫、工作區 ID、R2 endpoint／bucket，以及一致的 embedding 規格。
- 機器本地只需要程式、設定及可重建的模型快取；無須重新索引已完成建檔的文件。
- 本機 embedding 模型與 GPU 生成模型的 tokenizer 各自配套，不需互相相同。

文件與問題的 embedding 必須來自相容向量空間。實作固定模型 SHA、維度、正規化、最大 token 數及 query/document prompt，並以 fingerprint 拒絕混用。更換生成 LLM 不須重建索引；更換 embedding 則需要新工作區／索引及重新建檔。query 與 document 前綴可依模型規格不同。

## LoRA 生命周期

1. 使用訓練工具產生 adapter；本次未實作訓練。
2. 管理 CLI 驗證 adapter_config 與基礎模型，將 safetensors、設定及雜湊 manifest 上傳 R2，登錄 Neon。
3. GPU 主機從目錄選擇版本，下載並核對 SHA-256，取得固定基礎模型 revision 的 vLLM 啟動指令。
4. 操作者啟動 vLLM；前端由 `/v1/models` 確認登錄名稱已載入，再允許切換共用預設。
5. 每次問答固定模型快照並保存來源、prompt 版本、token usage 與延遲。

第一版狀態為 stored（已登錄、服務未載入）、available（服務列出名稱）、unknown（服務無法確認）。下載或啟動失敗由 CLI 顯示錯誤，尚無背景部署佇列與持久化 loading／failed 狀態。

LoRA 必須匹配基礎模型與 revision。若訓練設定未記錄 revision，登錄 CLI 要求操作者明確提供；檔案無法自行證明其訓練來源。`/v1/models` 僅確認服務名稱，不證明實際權重 SHA，需使用本專案固定版本的部署流程，且不得重用同名服務指向不同權重。

第一版只準備單一標準 LoRA，不做運行中卸載；切換到已載入模型只影響新請求。更換基礎模型或載入新的 adapter 需先等現有請求完成，再由操作者重啟 GPU 服務。後續可加入具權限與狀態追蹤的自動部署。

## 第一版可靠性界線

- PDF 以頁面文字及重疊字元窗口切段，尚未做語意段落或表格重建。
- 使用 pgvector cosine 精確搜尋，適合目前少量文件；混合關鍵字檢索尚待加入。
- 相似度門檻是可調起點，不能視為信心機率。
- 回答檢查 JSON 與引用 ID；可以拒絕捏造 ID，但不能證明每句話都受來源支持，仍需測試集驗證。
- 文件上傳及索引同步處理，DB 交易一次提交文件與全部片段；R2 成功但 DB 失敗可能留下未引用物件，重試會使用相同文件 object key。
- 同時上傳同內容可能重複計算 embedding，但資料庫唯一限制避免重複索引。第一版無背景工作重試及 OCR。
- 工作區是邏輯隔離，持有相同 DB 憑證的管理者可存取所有工作區；尚未實作資料庫 RLS／個人文件 ACL。

## 官方參考

- [Runpod 計價](https://www.runpod.io/pricing)
- [vLLM Qwen3.5 部署](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora/)
- [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)
- [Sentence Transformers 語意搜尋](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
