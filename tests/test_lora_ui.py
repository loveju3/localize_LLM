import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
from fastapi import UploadFile
from fastapi.testclient import TestClient
from app.main import app, authorize
from app.lora_ui import save_upload, load_adapter, MODEL, REVISION

class LoraUploadTests(unittest.TestCase):
    def test_wrong_names_rejected_before_storage(self):
        svc = MagicMock()
        with self.assertRaises(ValueError):
            save_upload(svc, 'test', UploadFile(filename='bad.json', file=BytesIO(b'{}')),
                        UploadFile(filename='adapter_model.safetensors', file=BytesIO(b'bad')))
        svc.store.put.assert_not_called()

    def test_upload_endpoint_requires_authentication(self):
        with TestClient(app) as client:
            self.assertIn(client.post('/api/lora', data={'model_id':'test'}).status_code, (401,503))
            self.assertIn(client.post('/api/lora/test/load').status_code, (401,503))

    def test_upload_endpoint_forwards_files_and_closes(self):
        app.dependency_overrides[authorize] = lambda: None
        try:
            with patch('app.main.service') as service, patch('app.main.lora_ui.save_upload', return_value={'id':'test'}) as save:
                with TestClient(app) as client:
                    response=client.post('/api/lora',data={'model_id':'test'},files={
                        'config_file':('adapter_config.json',b'{}'),
                        'weights_file':('adapter_model.safetensors',b'weights')})
                self.assertEqual(response.status_code,200)
                self.assertEqual(save.call_args.args[1],'test')
                self.assertTrue(save.call_args.args[2].file.closed)
        finally:
            app.dependency_overrides.clear()

    def test_incompatible_model_never_loaded(self):
        svc=MagicMock()
        svc.repo.model.return_value={'id':'bad','kind':'lora','base_model':MODEL,'base_revision':'a'*40}
        with self.assertRaises(ValueError): load_adapter(svc,'bad')
        svc.inference.served_models.assert_not_called()

    def test_already_available_is_idempotent(self):
        svc=MagicMock()
        svc.repo.model.return_value={'id':'test','kind':'lora','base_model':MODEL,'base_revision':REVISION,'served_name':'test','manifest':{'rank':8}}
        svc.settings.vllm_base_url='https://example.modal.run/v1'
        svc.inference.served_models.return_value={'test'}
        self.assertEqual(load_adapter(svc,'test')['status'],'available')
        svc.store.download_url.assert_not_called()
