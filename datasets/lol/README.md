# 英雄聯盟 Qwen3-8B LoRA 試驗資料

這是 **PC 版歷史更新的小規模 pilot**，不是完整遊戲百科，也不是 2026 最新攻略。模型自稱知識截止 2024-10 不等於已確認訓練截止日。

來源為 Riot 官方公告（自寫繁體中文問題與答案，非全文複製）：

- [14.22](https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-22-notes/)：Ambessa 上線與系統改動。
- [25.S1.1](https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-s1-1-notes/)：Atakhan、Fearless Draft、季節規則。
- [25.S1.2](https://www.leagueoflegends.com/en-ph/news/game-updates/patch-25-s1-2-notes/)：Mel 上線與版本調整。
- [25.06](https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-06-notes/)：保留為未見資料理解題。

核對日期：2026-10-05。公告可能含追加更新；答案僅對指定版本的列出事項成立。英雄名稱保留英文，避免地區譯名歧義。

## 檔案與實驗界線

| 檔案 | 筆數 | 用途 |
| --- | ---: | --- |
| `v1/train.jsonl` | 60 | 20 個事實各 3 個措辭，SFT＋LoRA 訓練 |
| `v1/validation.jsonl` | 20 | 同 20 個事實的新措辭，觀察 loss／選 checkpoint |
| `v1/test.jsonl` | 32 | 20 個記憶題＋8 個未見資料理解題＋4 個拒答題 |
| `v1/catalog.jsonl` | 112 | 網頁觀察用，含來源、版本、答案與分類；不可整份拿去訓練 |
| `v1/manifest.json` | — | 模型 revision、檔案筆數與 SHA-256 |

**記憶題刻意共享訓練事實，只保留不同問法**，用來測「是否學到這些更新」，不代表未見事實泛化。措辭變化較簡單，可能高估實際泛化能力。驗證集也共享事實，不能當獨立知識測試。

25.06 事實不出現在訓練／驗證；測試時會附短參考片段。它測的是依新資料回答的能力，不是憑空知道未訓練更新。部分片段很短且直接包含答案，屬基本閱讀控制題。拒答題另測資訊不足，不自動給語意分數。

60 筆含改寫而非 60 個獨立事實，適合流程驗證；正式結論需擴充資料、人工審核、獨立文件／主題切分與多次實驗。不要把 validation／test 的 assistant 答案混進訓練，也不要對測試題反覆調參。

## SFT 格式

每行一個 JSON 物件，只有 `messages`：

```json
{"messages":[{"role":"system","content":"版本限定的繁體中文助理規則"},{"role":"user","content":"請依 PC 版資料回答：Ambessa 在哪版推出？"},{"role":"assistant","content":"Ambessa 在 PC 版 14.22 推出。"}]}
```

使用 Qwen3 tokenizer 原生 chat template，不自行拼 ChatML 特殊 token。此 pilot 的推論使用 `enable_thinking=False`；訓練預處理也以相同模式套用 template，並檢查實際 token 與 assistant loss mask。不要將驗證用參考答案放進推論 prompt；未見資料題的 context 則本來就是輸入的一部分。

建議起點：LoRA rank 8 或 16、短序列先用 1024 token、1 epoch 後評估，再考慮增加。這些只是起點，不能保證效果；目前沒有執行訓練。訓練時先釋放同 GPU 的 vLLM 推論，避免記憶體不足。基礎模型 revision 固定為 manifest 中的版本，adapter 存檔後檢查相容性。

## 分頁觀察

啟動專案後開啟 `/experiments/lol`，輸入同一個 APP_API_KEY，連線載入模型。每頁 5 題，可切換 train／validation／test，每題有 A／B 模型回答、時間、輔助關鍵字檢查、人工評閱與備註。答案和來源預設收合。結果在頁面記憶體中，離開前匯出 JSON。

模型 A／B 不影響共用預設；只有推論端點實際已載入的模型能測試。尚無 adapter 時，兩欄選同一模型只是重複基線，不是 LoRA 比較。未部署訓練模型前不得宣稱提升。來源網址和參考答案不會隨記憶題送給模型。

重新產生：`.venv/bin/python datasets/lol/build.py`。測試：`.venv/bin/python -m unittest discover -s tests`。

完整基線（本機服務需已啟動，會呼叫真實 GPU，每次使用新檔名）：

```bash
.venv/bin/python scripts/lol_baseline.py --model qwen3-8b-b968826d --output data/lol/baseline-final-v1.jsonl
```

JSONL 保存 manifest 雜湊、實際輸入、模型版本、原始回答、耗時與待人工評閱欄位。`status=ok` 只表示 API 呼叫完成，並非答案正確。推論 temperature 目前沿用 0.2，正式比較應重複執行、固定生成設定並記錄變異。

2026-10-05 已在 Modal Qwen3-8B 完成最終題庫基線，報告在 `data/lol/baseline-final-v1.jsonl`：32 題呼叫成功、0 HTTP 錯誤；28 題有關鍵字檢查，其中 12 題符合、16 題未符合，4 題拒答未自動評分。這不是語意正確率。網頁第一題實測原模型答 Ambessa 在 13.10 推出，與官方 14.22 不符。尚未訓練 LoRA。

45 項單元測試執行成功，其中 6 項拋棄式 PostgreSQL 測試略過；新增測試檢查資料雜湊／筆數、prompt 分離、未見事實隔離、題目不洩漏推出版本、參考答案不送入記憶題推論、未載入模型拒絕測試，以及 API 驗證與分頁。
