# GPU 訓練頁面

開啟 http://127.0.0.1:8000/training，輸入工作區 API key 後連線。首頁與 LoRA 測試頁皆提供入口。

模型與資料固定為 Qwen3-8B 指定 revision 及英雄聯盟 v1，開放 rank 4／8／16／32、alpha、learning rate、epochs、max steps、max length、gradient accumulation、seed。使用「2 步短測試」先檢查 GPU 流程，再使用完整 epochs 設定；正數 max steps 優先於 epochs。

訓練提交到已部署的 Modal localize-llm-lora-training App，採用 detached FunctionCall，回傳 run ID。任務持續在雲端執行；頁面每 10 秒讀取已 commit 的 run.json／metrics.jsonl。資料在 Modal Volume，任務 handle 保存於本機 data/training/jobs；頁面重整／後端重啟後仍可查詢。提交與訓練失敗分別顯示失敗狀態，不用推測的進度冒充 GPU 訓練。

頁面呈現訓練與驗證 loss 曲線、逐步 metrics、實際 GPU、optimizer steps、耗時、參數比例、GPU tensor 峰值、設定、資料 SHA、套件版本與最佳 checkpoint。初始排程或映像／模型載入可能尚無數值。GPU 記憶體為 PyTorch tensor 峰值，不代表整張 GPU 用量；容器耗時不含排程和映像建立。

完成後可下載 run.json、metrics.jsonl、adapter_config.json、adapter_model.safetensors；也可點「將訓練結果加入測試模型」。後者從 Volume 下載、驗證 SHA、存入 R2 並登錄 Neon，之後在 A／B 測試頁選取及載入。既有模型 ID 不覆寫；相同產物重複加入可重用。

前端與 API 共用工作區 API key；持有者可提交會計費的 GPU 任務。本機需安裝 requirements-modal.txt 並登入 Modal；首次使用前部署 deploy/modal_train.py。GPU 任務最長一小時、最多一個容器、不自動重試，目前無頁面取消／自動續訓。可在 Modal 控制台停止任務。

目前訓練資料的 validation 與 train 共享事實，loss 降低不能視為測試正確率或泛化提升；需要在測試頁評閱回答。

## 實測紀錄

2026-10-05 已由 POST /api/training 提交 ui-ed755d970fd24241ab67，回傳 202 與 FunctionCall ID。NVIDIA L40S 完成 2 optimizer steps，驗證 loss 3.484849 → 3.062756，metrics 5 筆；四項下載均成功，adapter 雜湊與雲端 run.json 一致。新產物登錄與既有產物重複登錄皆通過。瀏覽器實際顯示完成狀態、數值、表格與 loss 曲線；亦可切換既有完整 15-step 訓練。58 項單元測試成功（6 項資料庫測試略過）。

網站使用 sessionStorage 保留同一瀏覽器分頁的工作區登入，切換回答測試／訓練／文件頁時自動連線。登出或關閉分頁可結束此會話；不使用 localStorage 永久保存金鑰。
