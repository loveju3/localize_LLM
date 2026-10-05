"""Build a small, version-scoped SFT pilot. No fetched article text is copied."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent / "v1"
SYSTEM = "你是英雄聯盟 PC 版知識助理，使用繁體中文簡潔回答。問題指定的版本優先；歷史版本不可說成目前版本。若提供參考資料，依資料回答；無法確定時明確說不知道，不捏造未公布的內容。"
URLS = {
    "14.22": "https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-22-notes/",
    "25.S1.1": "https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-s1-1-notes/",
    "25.S1.2": "https://www.leagueoflegends.com/en-ph/news/game-updates/patch-25-s1-2-notes/",
    "25.06": "https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-06-notes/",
}
# (version, question, authored reference answer, alternative keyword groups)
FACTS = [
 ("14.22", "Ambessa 在哪一個版本推出？", "Ambessa 在 PC 版 14.22 推出。", [["14.22"]]),
 ("14.22", "Ambessa 的上線日期與 UTC 時間是什麼？", "14.22 公告的上線時間是 2024-11-06 19:00 UTC。", [["2024"], ["11-06", "11月6", "11 月 6"], ["19:00"]]),
 ("14.22", "這個版本 ARAM 的特殊橋梁叫什麼？", "14.22 的 ARAM 特殊地圖是 Bridge of Progress。", [["Bridge of Progress", "進步之橋"]]),
 ("14.22", "計分板的賞金顯示門檻改成多少金幣？", "14.22 將賞金顯示門檻從 150 降到 100 金幣。", [["100"]]),
 ("25.S1.1", "2025 第一季的主題是哪個地區？", "25.S1.1 開始的 2025 第一季以 Noxus（諾克薩斯）為主題。", [["Noxus", "諾克薩斯"]]),
 ("25.S1.1", "新增的史詩野怪名稱是什麼？", "25.S1.1 新增 Atakhan。", [["Atakhan", "厄塔汗"]]),
 ("25.S1.1", "Atakhan 在遊戲幾分鐘出現？", "25.S1.1 的 Atakhan 在 20 分鐘出現。", [["20"]]),
 ("25.S1.1", "Atakhan 的坑在幾分鐘形成？", "25.S1.1 的 Atakhan 坑在 14 分鐘形成。", [["14"]]),
 ("25.S1.1", "Atakhan 的兩種形態英文名稱是什麼？", "25.S1.1 的兩種形態是 Ruinous 與 Voracious。", [["Ruinous"], ["Voracious"]]),
 ("25.S1.1", "預示者與巴龍的生成時間各是多少？", "25.S1.1 的 Rift Herald 在 16 分鐘、Baron Nashor 在 25 分鐘生成。", [["16"], ["25"]]),
 ("25.S1.1", "主堡防禦塔被摧毀後多久重生？", "25.S1.1 的主堡防禦塔被摧毀 3 分鐘後重生。", [["3", "三"]]),
 ("25.S1.1", "Fearless Draft 中一隻選過的英雄還能被另一隊選嗎？", "25.S1.1 說明的 Fearless Draft 中，系列賽選過的英雄在後續對局對兩隊都不可再選。", [["兩隊", "雙方"], ["不可", "不能"]]),
 ("25.S1.1", "取得 Feats of Strength 需要先完成幾項成就？", "25.S1.1 的 Feats of Strength 需要先完成三項中的兩項。", [["兩", "2"]]),
 ("25.S1.1", "Feats of Strength 的 Monster Slaying 需要幾個史詩野怪目標？", "25.S1.1 的 Monster Slaying 需要取得 3 個史詩野怪目標。", [["3", "三"]]),
 ("25.S1.1", "最早的砲車兵從第幾波改到第幾波？", "25.S1.1 將第一個砲車兵從第 3 波改到第 4 波。", [["3", "三"], ["4", "四"]]),
 ("25.S1.1", "Swiftplay 首批替換 Quickplay 的地區包含台灣嗎？", "包含；25.S1.1 公告將 Taiwan 列在首批替換地區中。", [["包含", "是", "有"], ["Taiwan", "台灣", "臺灣"]]),
 ("25.S1.2", "Mel 在哪個版本推出？", "Mel 在 PC 版 25.S1.2 推出。", [["25.S1.2"]]),
 ("25.S1.2", "Mel 的上線日期與 UTC 時間是什麼？", "25.S1.2 公告 Mel 在 2025-01-23 20:00 UTC 上線。", [["2025"], ["01-23", "1月23", "1 月 23"], ["20:00"]]),
 ("25.S1.2", "Cassiopeia 勝利成就被動每級移速從多少改到多少？", "25.S1.2 將 Triumphant 被動每級移速從 6 降到 5；一般被動未改。", [["6"], ["5"]]),
 ("25.S1.2", "Voracious Atakhan 的 Withdraw 觸發給擊殺方的金幣如何改動？", "25.S1.2 將 Withdraw 觸發給擊殺方的金幣由 100 提高至 200。", [["100"], ["200"]]),
]
HELDOUT = [
 ("25.06", "英雄賞金抑制的金錢領先偵測從幾分鐘改到幾分鐘？", "25.06 從 14 分鐘改到 6 分鐘。", [["14"], ["6"]]),
 ("25.06", "英雄賞金抑制提早後，目標賞金時間也變成 6 分鐘嗎？", "沒有；25.06 的目標賞金時間仍為 14 分鐘。", [["14"], ["沒有", "不", "仍"]]),
 ("25.06", "反換線規則開始時間改成多少？", "25.06 改為 1:35。", [["1:35"]]),
 ("25.06", "反換線規則在上路幾分幾秒結束？", "25.06 上路在 3:00 結束。", [["3:00"]]),
 ("25.06", "反換線規則在中路幾分幾秒結束？", "25.06 中路在 2:15 結束。", [["2:15"]]),
 ("25.06", "反換線的小兵金錢與經驗懲罰如何調整？", "25.06 從 50% 降為 25%。", [["50"], ["25"]]),
 ("25.06", "Atakhan 形態判定門檻如何調整？", "25.06 將形態門檻提高約 10%。", [["10"], ["提高", "增加", "上調"]]),
 ("25.06", "Naafiri 的 W 和 R 在這版有什麼變動？", "25.06 的 Naafiri W 與 R 交換位置。", [["W"], ["R"], ["交換", "互換"]]),
]


def record(identifier, fact, split, suite, question, context=""):
    version, _, answer, terms = fact
    # Asking for a release patch must not reveal that patch in the question.
    asks_patch = "哪一個版本推出" in question or "哪個版本推出" in question
    prompt = ("請依英雄聯盟 PC 版資料回答：" if asks_patch else
              f"請依英雄聯盟 PC 版 {version} 回答：") + question
    if context:
        prompt += "\n參考資料（改寫自 Riot 官方公告）：\n" + context
    return {"id": identifier, "split": split, "suite": suite,
            "fact_group": identifier.split('-')[0], "patch": version,
            "source_url": URLS[version], "checked_on": "2026-10-05",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "reference_answer": answer, "keyword_groups": terms}


def write(name, rows):
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    (ROOT / name).write_text(text, encoding="utf-8")
    return {"rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def main():
    ROOT.mkdir(exist_ok=True)
    catalog, train, validation = [], [], []
    for i, fact in enumerate(FACTS, 1):
        for j, prefix in enumerate(["", "請簡短說明：", "我想確認版本資料："], 1):
            row = record(f"f{i:02d}-train{j}", fact, "train", "knowledge_recall", prefix + fact[1])
            catalog.append(row)
            train.append({"messages": [*row["messages"], {"role": "assistant", "content": row["reference_answer"]}]})
        row = record(f"f{i:02d}-val", fact, "validation", "knowledge_recall", "核對歷史規則，" + fact[1])
        catalog.append(row)
        validation.append({"messages": [*row["messages"], {"role": "assistant", "content": row["reference_answer"]}]})
        catalog.append(record(f"f{i:02d}-test", fact, "test", "knowledge_recall", "朋友說他不確定，請幫忙回答：" + fact[1]))
    for i, fact in enumerate(HELDOUT, 1):
        catalog.append(record(f"h{i:02d}-test", fact, "test", "unseen_context", fact[1], fact[2]))
    for i, question in enumerate([
        "請給出 2030 年世界賽冠軍及比分。", "請給出尚未公布的 2031 年新英雄完整技能數值。",
        "只根據這段資料，Mel 在 25.S1.2 的全球勝率精確到小數點後兩位是多少？",
        "只根據這段資料，Ambessa 在 14.22 的全球選用率是多少？",
    ], 1):
        fact = ("25.06", question, "不知道；提供的資料未包含這項資訊，無法可靠回答。", [])
        row = record(f"u{i:02d}-test", fact, "test", "refusal", question, "資料僅說明 25.06 的反換線與賞金調整，未包含賽事結果或英雄勝率。")
        row["expected_refusal"] = True
        catalog.append(row)
    manifest = {"version": "lol-pilot-v1", "base_model": "Qwen/Qwen3-8B",
                "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "files": {"train.jsonl": write("train.jsonl", train),
                          "validation.jsonl": write("validation.jsonl", validation),
                          "test.jsonl": write("test.jsonl", [r for r in catalog if r["split"] == "test"]),
                          "catalog.jsonl": write("catalog.jsonl", catalog)}}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
