# Skill 與文件搜尋 Tool

在 `/documents` 的「與模型對話」勾選「依據文件回答」，選擇文件（未選取表示工作區全部文件），再送出問題。切換模式或文件範圍會清空歷史；回答下方可展開引用的檔名、PDF 頁序與原文。此模式只回答有文件證據的問題，一般閒聊請取消勾選。

## 實作對照

- `skills/document-qa/SKILL.md`：文件問答作業流程，包含補查、證據不足與 JSON 輸出規則。
- `app/skills.py`：僅載入應用程式允許的 Skill，不接受使用者或模型指定路徑。這是應用程式 Skill，並非自動安裝至 Codex 的 Skill。
- `app/tools.py`：`search_documents(question)` 的 schema 與執行器，重用 `RagService.search()`。文件範圍由 API 傳入，模型無法用工具參數覆寫。既有 Repository 保留工作區篩選。
- `app/inference.py`：傳送工具定義、解析工具呼叫，將呼叫 ID 與結果交回模型。
- `app/service.py`：每次問答最多兩次工具要求（錯誤要求也計數），最多三次推論。用 chunk ID 去重並統一來源編號，累計 token 用量。超額批次不執行，要求停止後仍呼叫工具則回傳上限訊息。
- `app/main.py`：`POST /api/chat` 接受 `mode: "document"`、可選的 UUID 陣列 `document_ids`；省略 mode 維持原有直接聊天。

API 範例：

```json
{
  "mode": "document",
  "document_ids": null,
  "messages": [{"role": "user", "content": "手冊中的保固期限多久？"}]
}
```

回應包含 `answer`、`citations`、`insufficient_evidence`、`skill`、`tool_trace`、`stop_reason`、`model`、`usage`、`elapsed_seconds`。聊天與工具軌跡不持久化；每輪重新取得證據，先前答案不能充當本輪來源。引用 ID 驗證不等於語意正確性驗證。

## vLLM 部署需求

Qwen3-8B 的伺服器需啟用 `--enable-auto-tool-choice --tool-call-parser hermes`。已加入 `deploy/modal_vllm.py` 的啟動參數；既有遠端服務需重新部署才會生效。RunPod 需在原有啟動命令加上相同參數。其他模型請依模型與 vLLM 版本選擇 parser。

參考：[vLLM 工具呼叫文件](https://docs.vllm.ai/en/stable/features/tool_calling/)。此修改不會自行重啟或重新部署遠端 GPU。

## 驗證

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_document_chat.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

2026-10-12：新增 13 項測試通過；完整測試 71 項，65 通過、6 跳過。涵蓋真實 FastAPI 路由至 httpx 模擬推論端點的工具往返、文件範圍、兩次搜尋的去重與編號、未知工具、錯誤參數修正、呼叫上限、缺乏證據及捏造來源。Skill 格式與套件依賴檢查通過。

本機瀏覽器另使用合成文件與替代推論服務確認：選取文件、送出問題、顯示答案／頁码、展開原文、切換模式清除歷史，以及一般聊天回覆。測試截圖保存在忽略 Git 的 `data/document-chat-validation/ui.png`。此測試沒有上傳文件至遠端或呼叫 GPU。

已設定的雲端 `/models` 探測逾時，因此尚未驗證真實 vLLM 工具輸出、Neon 檢索與模型回答品質。上述自動測試使用替代雲端服務，不代表雲端端到端驗證完成。
