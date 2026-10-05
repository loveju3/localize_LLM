# 英雄聯盟 SFT＋LoRA 訓練與 metrics

訓練工作獨立於現有 Modal 推論 App，不會改寫基礎模型、部署中的服務或共用預設模型。目前實作標準 BF16 LoRA，不是 QLoRA，也沒有 DPO。

## 設定與資料

`training/lora.py` 使用 Transformers Trainer＋PEFT，固定 Qwen3-8B revision。預設 rank 8、alpha 16、learning rate 1e-4、1 epoch、microbatch 1、gradient accumulation 4、sequence limit 1024、seed 42。attention／MLP 的七種 projection 層套用 LoRA，其餘參數凍結，程式會檢查沒有意外訓練基礎參數。

只讀 `train.jsonl` 與 `validation.jsonl`，不讀 `test.jsonl`。原生 Qwen3 chat template 使用 `enable_thinking=False`，system／user／生成提示 token 的 labels 為 -100，僅對 assistant completion 與結束 token 算 loss。prefix 不一致、超長或沒有回答 token 時直接拒絕訓練，避免靜默截斷或錯誤遮罩。

這份 validation 與訓練共享 20 個事實，eval loss 主要是同事實的新問法，不能解讀為未見知識泛化。資料僅 60 個改寫樣本，正式效果需要更多獨立事實與人工審核。

## Modal 執行

本機只需既有 Modal SDK；不要把 `requirements-training.txt` 裝進 Mac 的應用環境。GPU runtime 由 Modal 建立。

先跑短 smoke（2 optimizer steps，包含訓練前後 validation）：

```bash
.venv/bin/python -m modal run --env main --profile loveju3vup deploy/modal_train.py --run-id lol-smoke-001 --smoke
```

完整一輪：

```bash
.venv/bin/python -m modal run --env main --profile loveju3vup deploy/modal_train.py --run-id lol-r8-001 --rank 8 --epochs 1
```

每次 run ID 必須唯一，不覆寫既有結果。訓練最多一張 L40S（備選 A100-40GB）、CPU 4、RAM 32GiB、最長 3600 秒，不自動重試。這是 `modal run` 任務，完成後退出，不是常駐服務。訓練與推論若同時執行，會使用不同 GPU 資源並各自計費；可先停止推論 App 以節省費用。

模型快取沿用 `localize-llm-model-cache`；輸出寫入 `localize-llm-training-runs/<run-id>`，每次 metric 和 checkpoint 保存後 commit。Volume 儲存費另計。無 R2／Neon 金鑰傳給訓練容器。首次映像建立與模型下載會額外花時間。

## 追蹤項目

| 檔案／metric | 意義 |
| --- | --- |
| `run.json` | 狀態、設定、資料 SHA-256、套件版本、實際 GPU、可訓練參數比例、初始／最終 eval loss、adapter 雜湊 |
| `metrics.jsonl` | 每次 log 的 step、epoch、loss、learning rate、grad_norm、elapsed_seconds、GPU 記憶體 |
| `eval_loss` | 驗證集 assistant token 的交叉熵，越低通常越能模仿資料 |
| `eval_perplexity` | exp(eval_loss)，與此 loss 同源，不是另一個獨立品質評分 |
| `gpu_peak_allocated_gib` | PyTorch tensor 記憶體峰值；不等於 nvidia-smi 整台 GPU 用量 |
| `tensorboard/` | loss／eval loss／learning rate 等 Trainer scalars |
| `checkpoints/` | 每 5 steps 保存，最多 2 份；完成後 adapter 使用最佳已保存 eval-loss checkpoint，若 smoke 未達保存步數則使用最後權重 |
| `adapter/` | 標準 adapter_config.json＋adapter_model.safetensors，附完整 revision，接得上既有上傳驗證 |

失敗會記錄 `status=failed` 與 error_type，不把失敗當完成。任務遭平台強制中止時，最後一次 commit 的 metrics／checkpoint 才是可恢復資料；目前尚未提供自動 resume。

下載產物（依 run ID 調整，先建立父目錄；CLI 會在父目錄下建立同名 run 資料夾）：

```bash
mkdir -p data/training
.venv/bin/python -m modal volume get localize-llm-training-runs /lol-r8-001 data/training --env main --profile loveju3vup
```

另建獨立 TensorBoard 環境或在 GPU runtime 中使用已安裝的 TensorBoard：

```bash
tensorboard --logdir data/training --host 127.0.0.1 --port 6006
```

開啟 `http://127.0.0.1:6006`。JSONL 也可直接分析，不依賴第三方追蹤帳戶。

## 訓練後品質 metrics

先以既有 `app.cli upload-lora` 登錄／保存 adapter，再部署啟用 LoRA 的 vLLM（目前 Modal 基礎推論部署尚未自動載入 adapter）。在英雄聯盟分頁選 A＝原模型、B＝實際載入的 LoRA，使用相同 test 資料與生成設定。

分別統計記憶題、未見資料理解題、拒答題的人工正確率，以及過度拒答、關鍵字符合率、生成耗時。已有基線在 `data/lol/baseline-final-v1.jsonl`。不能以 loss 下降或關鍵字符合率直接宣稱答案品質改善；test 只做最後評估，不用來選 checkpoint。

參考：[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)、[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)。

## 本次驗證範圍

已用固定 revision 的真實 Qwen3 tokenizer 驗證 80 筆 train／validation：最長 157 tokens，assistant labels 不含 system／user，並保留結束 token。token 邊界採 fast tokenizer offsets 判定，跨界非空白 token 直接拒絕。47 項單元測試通過，其中 6 項資料庫測試略過。

2026-10-05 已在 Modal NVIDIA L40S 實際完成兩次訓練，套件為 torch 2.8.0+cu128、transformers 4.57.1、peft 0.17.1：

| Run ID | optimizer steps | 初始 eval loss | 最終 eval loss | 訓練平均 loss | GPU tensor 峰值 |
| --- | --- | --- | --- | --- | --- |
| lol-smoke-20261005-01 | 2 | 3.484849 | 3.054699 | 2.980455 | 15.90 GiB |
| lol-r8-20261005-01 | 15（1 epoch） | 3.484849 | 1.230207 | 1.762941 | 15.91 GiB |

完整訓練採用最佳 checkpoint-15，訓練迴圈耗時 46.76 秒，容器內流程共 68.36 秒（不含映像建立及排程等待）。可訓練參數 21,823,488，約佔總參數 0.266%。

兩份產物都下載到 `data/training/<run-id>/`，含 run.json、metrics.jsonl、TensorBoard events、adapter 與 tokenizer；完整訓練另有 checkpoint-10／15。下載後檢查所有 adapter 檔案 SHA-256 與 run.json 一致、所有權重有限、252 個原本零初始化的 LoRA B 矩陣皆含非零值，以及 metrics 的 optimizer steps 連續完整。檢查結果保存在各 run 的 verification.json，可確認真正更新了 LoRA 權重。

完整訓練記錄：[Modal run](https://modal.com/apps/loveju3vup/main/ap-z5MLfFOMTFxzzYKzwrcK8D)。兩個 modal run 任務均已正常退出；訓練 Volume 保留產物。adapter 已透過頁面使用的 API 上傳並載入 Modal 推論服務，模型 ID 為 lol-r8-20261005-01。第一題 Ambessa 的 A／B 測試兩者皆錯（原模型 13.10、LoRA 12.10；參考 14.22）；loss 下降不能視為回答品質改善。結果保存在 serving-verification.json。
