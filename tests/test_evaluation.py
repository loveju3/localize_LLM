import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.evaluation import EvaluationCase, load_cases, main, run_evaluation, score


DOC = "00000000-0000-0000-0000-000000000001"


def case(**changes):
    values = {"id": "q1", "category": "lookup", "question": "營收？", "expected_refusal": False,
              "expected_sources": [{"document_id": DOC, "page": 2}], "answer_contains": ["20%"]}
    values.update(changes)
    return EvaluationCase.model_validate(values)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rag = MagicMock()
        self.rag.settings = Settings(_env_file=None, embedding_revision="a" * 40)
        self.rag.repo.model.side_effect = lambda name: {
            "id": name, "kind": "base", "base_model": "test/model", "base_revision": "b" * 40,
            "served_name": name, "artifact_prefix": None}
        self.rag.inference.served_models.return_value = {"base", "comparison"}
        self.sources = [{"source_id": "S1", "document_id": DOC, "page": 2, "text": "Revenue grew 20%."}]
        self.rag.search.return_value = self.sources
        self.rag.inference.answer.return_value = (json.dumps({
            "answer": "20%", "source_ids": ["S1"], "insufficient_evidence": False}), {"total_tokens": 24})
        self.records = []

    def run_cases(self, cases, models=None):
        return run_evaluation(cases, self.rag, models or ["base"], self.records.append)

    def test_models_share_retrieval_without_changing_active_model(self):
        summary = self.run_cases([case()], ["base", "comparison"])
        self.rag.search.assert_called_once()
        self.rag.repo.activate.assert_not_called()
        self.rag.repo.save_run.assert_not_called()
        self.assertEqual(self.rag.inference.answer.call_count, 2)
        self.assertIs(self.rag.inference.answer.call_args_list[0].args[1],
                      self.rag.inference.answer.call_args_list[1].args[1])
        self.assertEqual(summary["comparison"]["metrics"]["citation_page_recall"]["mean"], 1)
        self.assertEqual(self.records[1]["usage"]["total_tokens"], 24)
        self.assertIsNone(self.records[1]["manual_review"]["answer_correct"])

    def test_empty_retrieval_is_refusal_without_inference(self):
        self.rag.search.return_value = []
        summary = self.run_cases([case(expected_refusal=True, expected_sources=[], answer_contains=[])])
        self.rag.inference.answer.assert_not_called()
        self.assertEqual(summary["base"]["metrics"]["refusal_correct"], {"mean": 1, "count": 1})
        self.assertEqual(summary["base"]["metrics"]["citation_page_recall"], {"mean": None, "count": 0})

    def test_failure_is_not_counted_as_refusal_and_later_cases_run(self):
        good = self.rag.inference.answer.return_value
        self.rag.inference.answer.side_effect = [RuntimeError("secret-key"), good]
        summary = self.run_cases([case(), case(id="q2")])
        self.assertEqual(summary["base"]["errors"], 1)
        self.assertEqual(summary["base"]["completed"], 1)
        self.assertEqual(summary["base"]["metrics"]["refusal_correct"]["count"], 1)
        self.assertNotIn("secret-key", json.dumps(self.records))

    def test_retrieval_failure_skips_models_and_continues(self):
        self.rag.search.side_effect = [RuntimeError("database-password"), self.sources]
        summary = self.run_cases([case(), case(id="q2")], ["base", "comparison"])
        self.assertEqual(summary["base"]["errors"], 1)
        self.assertEqual(self.rag.inference.answer.call_count, 2)
        self.assertEqual(self.records[1]["error_stage"], "retrieval")
        self.assertNotIn("database-password", json.dumps(self.records))

    def test_unloaded_model_fails_before_retrieval(self):
        with self.assertRaises(ValueError):
            self.run_cases([case()], ["absent"])
        self.rag.search.assert_not_called()

    def test_invalid_citation_is_refused_and_raw_output_retained(self):
        self.rag.inference.answer.return_value = ('{"source_ids":["S999"]}', {})
        summary = self.run_cases([case()])
        self.assertEqual(summary["base"]["metrics"]["refusal_correct"]["mean"], 0)
        self.assertEqual(summary["base"]["metrics"]["citation_page_recall"]["mean"], 0)
        self.assertIn("S999", self.records[1]["raw_answer"])

    def test_recall_counts_distinct_document_pages(self):
        expected = [{"document_id": DOC, "page": 2}, {"document_id": DOC, "page": 3}]
        result = {"answer": "20%", "citations": self.sources * 2, "insufficient_evidence": False}
        metrics = score(case(expected_sources=expected), self.sources * 2, result)
        self.assertEqual(metrics["retrieval_page_recall"], .5)
        self.assertEqual(metrics["citation_page_recall"], .5)

    def test_document_filter_is_passed_to_retrieval(self):
        scoped = case(document_ids=[DOC])
        self.run_cases([scoped])
        self.rag.search.assert_called_once_with(scoped.question, scoped.document_ids)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "cases.jsonl")
            serialized = case().model_dump_json()
            path.write_text(serialized + "\n", encoding="utf-8")
            cases, digest = load_cases(path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(digest), 64)
            for invalid in ("", serialized + "\n" + serialized, "{}", "not-json"):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)
        for changes in ({"question": " "}, {"expected_refusal": "false"},
                        {"answer_contains": [" "]}, {"expected_refusal": True},
                        {"document_ids": []}, {"unexpected": True}):
            with self.assertRaises(ValueError):
                case(**changes)

    def test_validate_only_never_initializes_services(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "cases.jsonl")
            path.write_text(case().model_dump_json(), encoding="utf-8")
            with patch("app.main.service") as factory, patch("builtins.print"):
                self.assertEqual(main([str(path), "--validate-only"]), 0)
                factory.assert_not_called()

    def test_cli_report_is_checkpointed_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, output = Path(directory, "cases.jsonl"), Path(directory, "run.jsonl")
            dataset.write_text(case().model_dump_json(), encoding="utf-8")
            args = [str(dataset), "--model", "base", "--output", str(output)]
            with patch("app.main.service", return_value=self.rag), patch("builtins.print"):
                self.assertEqual(main(args), 0)
                original = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    main(args)
                self.assertEqual(output.read_bytes(), original)
            records = [json.loads(line) for line in original.splitlines()]
            self.assertEqual([r["type"] for r in records], ["header", "configuration", "result", "summary"])

    def test_cli_fatal_error_is_redacted_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, output = Path(directory, "cases.jsonl"), Path(directory, "run.jsonl")
            dataset.write_text(case().model_dump_json(), encoding="utf-8")
            with patch("app.main.service", side_effect=RuntimeError("secret")), patch("builtins.print"):
                self.assertEqual(main([str(dataset), "--model", "base", "--output", str(output)]), 1)
            self.assertNotIn("secret", output.read_text())
            self.assertEqual(json.loads(output.read_text().splitlines()[-1])["type"], "fatal_error")
