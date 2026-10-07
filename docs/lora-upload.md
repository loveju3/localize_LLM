# 頁面上傳與測試 LoRA

開啟 `http://127.0.0.1:8000/experiments/lol`，輸入工作區 API key 並連線。

1. 在「加入訓練好的 LoRA」填入唯一模型 ID。
2. 選擇同一份訓練產物的 `adapter_config.json` 與 `adapter_model.safetensors`。目前不接受 ZIP 或 pickle 權重。
3. 點「上傳到雲端」。伺服器驗證格式與固定基礎模型版本，儲存到既有 R2，並在 Neon 登錄不可覆寫的模型版本。
4. 在「已上傳 LoRA」選擇模型，點「載入並選為 B 模型」。這會啟動 Modal GPU、把驗證過雜湊的檔案放到 `localize-llm-adapters` Volume，載入 vLLM，並確認 `/v1/models` 已列出它。
5. A 選原模型、B 選 LoRA，對同一題分別執行並評閱；共用預設模型不會被更改。

支援固定 revision 的 Qwen/Qwen3-8B、標準 LoRA、rank 上限 64、權重上限 2 GiB、設定檔上限 64 KiB。設定檔須明確包含 revision。ID 已登錄時請直接選擇載入，或使用新的 ID 上傳新版本。

Modal 冷啟動會掃描 Volume 內完整 adapter 並以 `--lora-modules` 載入。容器运行時新增的 adapter，透過已驗證 API key 的代理 reload Volume，再呼叫 vLLM 動態載入接口。代理只開放模型列表、對話與受限制的 adapter 載入；GPU 路徑由伺服器決定，瀏覽器不取得 R2／Modal／vLLM 金鑰。上傳儲存成功與 GPU 載入成功是分開的；載入失敗後保留雲端產物，可以重試。

本機需安裝 `requirements-modal.txt`，並先完成 Modal SDK 登入。自動載入目前只支援 Modal，RunPod 可沿用 CLI prepare-model 部署。基礎推論仍支援兩個供應商。

目前測試页適用個人信任工作區；持有工作區 API key 者可以上傳並載入模型。多使用者服務應另外分離管理員權限。載入與對話使用 GPU；持久儲存費用依雲端服務計算。

參考：[vLLM LoRA](https://docs.vllm.ai/en/stable/features/lora/)。

## 實測

2026-10-05：透過本機 POST /api/lora 上傳 83 MiB adapter，回傳 200；POST /api/lora/lol-r8-20261005-01/load 回傳 available。模型列表確認原模型與 adapter 都 available，同一題 A／B 請求皆成功，結果在 data/training/lol-r8-20261005-01/serving-verification.json。此題兩者皆錯，不能宣稱訓練品質改善。52 項單元測試執行成功（6 項資料庫測試略過）。

另已驗證閒置縮到零後的 GPU 冷啟動：容器重新載入基礎權重及持久 adapter，重複載入接口回傳 available，原模型與 LoRA 均列為 available。

網站使用 sessionStorage 保留同一瀏覽器分頁的工作區登入，切換回答測試／訓練／文件頁時自動連線。登出或關閉分頁可結束此會話；不使用 localStorage 永久保存金鑰。
