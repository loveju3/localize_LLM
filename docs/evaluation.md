# 文件問答評估

使用 `python -m app.evaluation` 批次評估已建檔的 PDF。每一題只檢索一次，再讓指定模型使用相同片段回答；不更動共用預設模型，也不寫入一般問答紀錄。結果保存在本機 JSONL 報告。

## 1. 準備題庫（本機，不需要 GPU）

複製 `docs/evaluation.example.jsonl` 到 `data/evaluations/questions.jsonl`，依實際 PDF 改寫。範本只有四種題型示例，UUID、頁碼、答案皆為占位內容，不是可用的品質基準。正式評估依需求準備 20～30 題，涵蓋直接查找、跨段整理、表格數字、無答案，中英文都應包含。

每行一個 JSON 物件：

| 欄位 | 意義 |
| --- | --- |
| `id` | 題庫內唯一的題目 ID |
| `category` | 題型，自訂文字 |
| `question` | 非空問題，最多 2000 字元 |
| `document_ids` | 限定檢索文件的 UUID 清單；省略代表整個工作區，不接受空清單 |
| `expected_refusal` | 必填 JSON 布林值，預期是否應拒答 |
| `expected_sources` | 預期來源的 `document_id` 與 `page`；page 是從 1 起算的 PDF 頁序 |
| `answer_contains` | 選填，答案應包含的字串；忽略大小寫，所有字串都需出現 |
| `reference_answer` | 人工核對用參考答案，不傳給模型 |

文件 UUID 可從已授權的 `GET /api/documents` 回應取得。拒答題不填 expected_sources 與 answer_contains。預期來源必須在指定文件範圍內。同頁多個片段只計一次；若同一事實可從不同頁面找到，請自行審核替代引用，工具目前把列出的每個不同頁面都視為應命中目標。

先做離線格式檢查，不讀雲端服務、不下載 embedding：

```bash
.venv/bin/python -m app.evaluation docs/evaluation.example.jsonl --validate-only
.venv/bin/python -m app.evaluation data/evaluations/questions.jsonl --validate-only
```

## 2. 執行真實評估（Mac 執行，RunPod 提供推論）

依 getting-started.md 完成 Neon／R2 工作區設定、PDF 建檔、embedding 固定版本與 vLLM 連線。模型須已登錄，且出現在 vLLM `/v1/models` 中。評估 CLI 直接使用 `.env` 與本機服務類別，不必先啟動 Uvicorn。

```bash
.venv/bin/python -m app.evaluation data/evaluations/questions.jsonl \
  --model base-v1 \
  --output data/evaluations/base-v1-run1.jsonl
```

若多個模型已由同一 vLLM endpoint 提供，可重複指定 `--model`，例如 `--model base-v1 --model adapter-v1`。工具逐題、逐模型依序執行。不同基礎模型若必須分開重啟 GPU，需分次執行；跨次執行會重新檢索，因此比較前須確認題庫、索引與檢索設定一致，並核對報告中的實際片段。工具目前不支援跨次載入凍結片段。

執行會使用雲端推論，可能產生 GPU 費用；本機尚無 embedding 快取時，也會下載設定的固定版本模型。無檢索片段的題目直接產生拒答，不呼叫生成模型。

## 3. 閱讀報告

報告依序包含 `header`、`configuration`、每題每模型的 `result`、最後的 `summary`。每筆立即 flush；若中途停止，已完成結果仍可讀取，但可能沒有總結。輸出路徑必須是新檔案，避免覆蓋既有實驗。不支援續跑，自行重新執行時使用新檔名。

- 保存題庫 SHA-256、模型版本、embedding fingerprint、檢索設定、prompt 版本與雜湊。
- 保存原問題、預期答案、實際檢索片段、原始模型輸出、驗證後答案、引用、token usage、檢索與推論耗時。
- `refusal_correct`：實際拒答是否符合預期。格式或引用驗證失敗也會被應用轉成拒答；需查看 `raw_answer` 區分主動拒答與格式失敗。
- `retrieval_page_recall`：預期文件頁面中有多少出現在檢索結果。
- `citation_page_recall`：預期文件頁面中有多少被答案引用。
- `answer_terms_present`：是否包含所有指定字串，僅為字面檢查。

未指定預期頁面或關鍵字時，相應分數為 `null`，不列入該指標平均值；每個平均值附有實際分母 `count`。連線與執行失敗另外記為 `error`，不當成正確拒答或成功測試。遇單題錯誤會繼續後續題目，CLI 最後回傳非零狀態；設定／模型預檢失敗則寫入 `fatal_error` 並結束。

以上自動指標不證明答案語意正確。請逐筆填寫 `manual_review.answer_correct`、`citations_supported`（true／false）及 `notes`；特別核對表格單位、年份、數字與跨段結論。工具不自動合併人工分數，也不設任意「通過」門檻。成功結束只代表批次執行完成。

報告含文件原文及模型輸出，放在已排除 Git 的 `data/` 下。費用請另外記錄 RunPod 帳單、啟停時間與儲存成本；token 數與請求耗時不能直接當作 GPU 租用費用。

## 目前驗證界線

模擬測試驗證題庫檢查、固定片段比較、文件範圍、指標計算、拒答、錯誤處理與報告保存。尚未用真實 PDF 題庫、Neon 或 RunPod 執行品質評估；需要實際文件才能建立有效的 20～30 題基準。
