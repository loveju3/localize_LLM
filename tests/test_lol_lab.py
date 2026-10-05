import hashlib
import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app import lol_lab
from app.main import app, settings, service
from app.config import Settings


class LolLabTests(unittest.TestCase):
    def test_dataset_integrity_and_split_contract(self):
        manifest = json.loads((lol_lab.DATA / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            raw = (lol_lab.DATA / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
            self.assertEqual(len(raw.splitlines()), expected["rows"])
        rows = lol_lab.cases()
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        train = [json.loads(x) for x in (lol_lab.DATA / "train.jsonl").read_text().splitlines()]
        self.assertTrue(all([m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in train))
        self.assertTrue(all("參考資料" not in r["messages"][1]["content"] for r in train))
        self.assertFalse(any("25.06" in r["messages"][1]["content"] for r in train))
        for row in rows:
            if row['fact_group'] in ('f01', 'f17'):
                self.assertNotIn(row['patch'], row['messages'][1]['content'])
        prompts = [{r["messages"][1]["content"] for r in rows if r["split"] == s}
                   for s in ("train", "validation", "test")]
        self.assertFalse(prompts[0] & prompts[1] or prompts[0] & prompts[2] or prompts[1] & prompts[2])

    def test_reference_not_sent_and_default_not_changed(self):
        svc = MagicMock()
        svc.repo.model.return_value = {"id": "base", "base_model": "Qwen/Qwen3-8B",
            "base_revision": "a" * 40, "served_name": "Qwen/Qwen3-8B"}
        svc.inference.served_models.return_value = {"Qwen/Qwen3-8B"}
        svc.inference.chat.return_value = ("14.22", {})
        result = lol_lab.run_case(svc, "f01-test", "base")
        sent = svc.inference.chat.call_args.args[0]
        self.assertEqual([m["role"] for m in sent], ["user"])
        self.assertNotIn(result["reference_answer"], str(sent))
        self.assertTrue(result["keyword_match"])
        svc.repo.activate.assert_not_called()
        svc.inference.chat.reset_mock()
        svc.inference.served_models.return_value = set()
        with self.assertRaises(ValueError):
            lol_lab.run_case(svc, "f01-test", "base")
        svc.inference.chat.assert_not_called()

    def test_api_auth_paging_and_server_selected_case(self):
        settings_patch = patch('app.main.settings', return_value=Settings(_env_file=None, app_api_key="test-key"))
        settings_patch.start()
        try:
            with TestClient(app) as client:
                self.assertEqual(client.get('/api/lol/cases').status_code, 401)
                headers = {"Authorization": "Bearer test-key"}
                first = client.get('/api/lol/cases', headers=headers).json()
                second = client.get('/api/lol/cases?page=2', headers=headers).json()
                self.assertEqual(first['total'], 32)
                self.assertFalse({r['id'] for r in first['cases']} & {r['id'] for r in second['cases']})
                self.assertEqual(client.get('/api/lol/cases?page=0', headers=headers).status_code, 400)
                with patch('app.main.lol_lab.run_case', side_effect=LookupError('題目不存在')):
                    app.dependency_overrides[service] = lambda: MagicMock()
                    self.assertEqual(client.post('/api/lol/run', headers=headers,
                        json={'case_id':'missing','model_id':'base'}).status_code, 404)
        finally:
            settings_patch.stop()
            app.dependency_overrides.clear()
