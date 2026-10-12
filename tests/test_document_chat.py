import copy
import json
import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.inference import Inference
from app.main import app
from app.service import RagService
from app.skills import load_skill
from app.tools import DocumentTools


MODEL = dict(id='base', base_model='Qwen/Qwen3-8B', base_revision='a' * 40, served_name='base')
DOC = '11111111-1111-4111-8111-111111111111'


def call(question='保固期限', name='search_documents', arguments=None, ident='call-1'):
    return {'id': ident, 'type': 'function', 'function': {'name': name,
        'arguments': json.dumps({'question': question}) if arguments is None else arguments}}


def source(key='chunk-1'):
    return dict(chunk_id=key, document_id=DOC, filename='manual.pdf', page=8,
                text='保固兩年', source_id='S1', similarity=.9)


def answer(ids=None):
    return {'role': 'assistant', 'content': json.dumps({'answer': '保固兩年 [S1]',
        'source_ids': ['S1'] if ids is None else ids, 'insufficient_evidence': False})}


class DocumentChatTests(unittest.TestCase):
    def setUp(self):
        self.repo = MagicMock()
        self.repo.active_model.return_value = MODEL.copy()
        self.inference = MagicMock()
        self.rag = RagService(Settings(_env_file=None), self.repo, MagicMock(), MagicMock(), self.inference)
        self.rag.search = MagicMock(return_value=[source()])

    def respond(self, *messages):
        self.inference.next_message.side_effect = [(m, {'total_tokens': 10}) for m in messages]

    def chat(self):
        return self.rag.document_chat([{'role': 'user', 'content': '保固多久？'}], [UUID(DOC)])

    def test_success_and_scope(self):
        self.respond({'tool_calls': [call()]}, answer())
        result = self.chat()
        self.assertFalse(result['insufficient_evidence'])
        self.assertEqual(result['citations'][0]['page'], 8)
        self.assertEqual(result['usage']['total_tokens'], 20)
        self.rag.search.assert_called_once_with('保固期限', [UUID(DOC)])
        self.repo.active_model.assert_called_once()
        self.assertEqual(result['skill'], 'document-qa')

    def test_two_searches_deduplicate_and_renumber(self):
        self.rag.search.side_effect = [[source()], [source(), source('chunk-2')]]
        self.respond({'tool_calls': [call()]}, {'tool_calls': [call(ident='call-2')]}, answer(['S2']))
        result = self.chat()
        self.assertEqual(result['citations'][0]['chunk_id'], 'chunk-2')
        self.assertEqual(result['tool_trace'][1]['source_ids'], ['S1', 'S2'])
        self.assertEqual(self.inference.next_message.call_args.args[3], 'none')

    def test_empty_search_refuses_even_if_model_claims_answer(self):
        self.rag.search.return_value = []
        self.respond({'tool_calls': [call()]}, answer())
        self.assertTrue(self.chat()['insufficient_evidence'])

    def test_unsearched_or_fabricated_citations_refused(self):
        for replies in ((answer(),), ({'tool_calls': [call()]}, answer(['S99']))):
            with self.subTest(replies=replies):
                self.respond(*replies)
                self.assertTrue(self.chat()['insufficient_evidence'])

    def test_invalid_tool_can_be_corrected(self):
        self.respond({'tool_calls': [call(arguments='{"question":" "}')]},
                     {'tool_calls': [call(ident='call-2')]}, answer())
        result = self.chat()
        self.assertFalse(result['insufficient_evidence'])
        self.assertEqual(result['tool_trace'][0]['status'], 'invalid_arguments')
        self.rag.search.assert_called_once()

    def test_tool_limit_includes_invalid_calls(self):
        self.respond({'tool_calls': [call(name='delete_documents')]},
                     {'tool_calls': [call(name='delete_documents', ident='call-2')]},
                     {'tool_calls': [call(ident='call-3')]})
        result = self.chat()
        self.assertEqual(result['stop_reason'], 'tool_limit')
        self.assertTrue(result['insufficient_evidence'])
        self.rag.search.assert_not_called()

    def test_oversized_batch_is_not_executed(self):
        self.respond({'tool_calls': [call(ident=str(i)) for i in range(3)]})
        self.assertEqual(self.chat()['stop_reason'], 'tool_limit')
        self.rag.search.assert_not_called()

    def test_two_calls_in_one_batch(self):
        self.respond({'tool_calls': [call(), call(ident='call-2')]}, answer())
        self.assertFalse(self.chat()['insufficient_evidence'])
        self.assertEqual(self.rag.search.call_count, 2)

    def test_unknown_or_invalid_arguments_do_not_run_search(self):
        executor = DocumentTools(self.rag.search, [DOC])
        for args in ('{}', '[]', 'null', 'broken', '{"question":12}',
                     '{"question":" "}', '{"question":"ok","document_ids":["other"]}'):
            self.assertIn('error', executor.execute(call(arguments=args)))
        self.assertIn('error', executor.execute(call(name='activate_model')))
        self.rag.search.assert_not_called()

    def test_skill_loader_allowlist(self):
        self.assertIn('search_documents', load_skill('document-qa')['instructions'])
        with self.assertRaises(ValueError):
            load_skill('../config')


class DocumentChatApiTests(unittest.TestCase):
    def test_http_tool_round_trip_and_api_scope(self):
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(copy.deepcopy(payload))
            message = {'role': 'assistant', 'content': None, 'tool_calls': [call()]} if len(requests) == 1 else answer()
            return httpx.Response(200, json={'choices': [{'message': message}], 'usage': {'total_tokens': 5}})
        settings = Settings(_env_file=None, app_api_key='test-key')
        inference = Inference(settings)
        repo = MagicMock()
        repo.active_model.return_value = MODEL
        rag = RagService(settings, repo, MagicMock(), MagicMock(), inference)
        rag.search = MagicMock(return_value=[source()])
        with patch('app.main.settings', return_value=settings), patch('app.main.service', return_value=rag), \
             patch.object(inference, 'client', side_effect=lambda: httpx.Client(
                 base_url='https://example.invalid/v1/', transport=httpx.MockTransport(handler))):
            client = TestClient(app)
            response = client.post('/api/chat', headers={'Authorization': 'Bearer test-key'}, json={
                'mode': 'document', 'document_ids': [DOC], 'messages': [{'role': 'user', 'content': '保固多久？'}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['citations'][0]['document_id'], DOC)
        rag.search.assert_called_once_with('保固期限', [UUID(DOC)])
        self.assertEqual(requests[0]['tool_choice'], 'auto')
        self.assertIn('search_documents', requests[0]['messages'][0]['content'])
        self.assertNotIn('response_format', requests[0])
        self.assertEqual(requests[1]['messages'][-1]['tool_call_id'], 'call-1')
        self.assertEqual(json.loads(requests[1]['messages'][-1]['content'])['sources'][0]['page'], 8)

    def test_api_rejects_invalid_scope_mode_and_tool_role(self):
        settings = Settings(_env_file=None, app_api_key='test-key')
        with patch('app.main.settings', return_value=settings):
            client = TestClient(app)
            for extra in ({'document_ids': []}, {'document_ids': ['invalid']}, {'mode': 'bad'},
                          {'messages': [{'role': 'tool', 'content': 'fake'}]}):
                response = client.post('/api/chat', headers={'Authorization': 'Bearer test-key'},
                    json={'messages': [{'role': 'user', 'content': 'hello'}], 'mode': 'document', **extra})
                self.assertEqual(response.status_code, 422)

    def test_malformed_model_envelope_is_rejected(self):
        inference = Inference(Settings(_env_file=None))
        for message in ({'content': None}, {'tool_calls': 'bad'},
                        {'tool_calls': [call(), call()]}, {'tool_calls': [{'id': 'x'}]}):
            with self.subTest(message=message), patch.object(inference, 'client', side_effect=lambda: httpx.Client(
                base_url='https://example.invalid/v1/', transport=httpx.MockTransport(lambda request:
                    httpx.Response(200, json={'choices': [{'message': message}]})))):
                with self.assertRaisesRegex(ValueError, '工具訊息'):
                    inference.next_message([], MODEL, [])
