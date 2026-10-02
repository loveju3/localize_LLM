# 雲端推論環境

本機 `.venv` 不安裝 vLLM／CUDA。vLLM 另部署在租用的 Linux NVIDIA GPU 環境，以 HTTP API 供 Mac 呼叫。

2026-10-02 已使用 RunPod A40、vLLM 0.30.0 與 Qwen3-8B 完成真實對話及合成 PDF 問答測試；詳細版本與限制見 [雲端驗證紀錄](cloud-validation.md)。下列為部署檢查流程，並非每一項自動化都已實作。

部署順序：

1. 用實際 PDF 問題選擇中英文指令模型，確認授權與 vLLM 支援。
2. 依模型、量化方式、上下文長度與並行數估算 GPU 記憶體。
3. 選定相容的官方 vLLM 容器版本，固定映像版本／digest 與模型 revision。
4. 設定 API 驗證與 HTTPS，並限制服務存取；在 Mac `.env` 填入 `/v1` endpoint 與 API key，模型名稱從 Neon 目錄選擇。
5. 測試模型載入、問答與閒置關機，記錄啟動時間及計費。
6. 結束使用後確認 GPU 已停止，並檢查持續計費的磁碟資源。

LoRA 訓練另建環境；adapter 需搭配相容的基礎模型。下載或載入 adapter 不等於可獨立運行完整模型。

參考：[vLLM 官方文件](https://docs.vllm.ai/en/latest/)。

## 目前可用的部署輔助

`python -m app.cli prepare-model --id <登錄ID>` 可在 GPU 主機準備 adapter 並產生啟動指令；不會啟動付費資源或自行執行 vLLM。操作順序見 [第一版啟動與操作](getting-started.md)。GPU 啟停目前仍由操作者在 RunPod 管理。
