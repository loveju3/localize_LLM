import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app, authorize
from app import training_ui

class TrainingPageTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[authorize]=lambda:None
        self.client=TestClient(app)
    def tearDown(self): app.dependency_overrides.clear()
    def test_fixed_identity_and_bounds(self):
        for data in [{'rank':64},{'max_steps':0},{'model':'Other/Model'},{'learning_rate':0},{'seed':-1}]:
            with patch('app.main.training_ui.start') as start:
                # max_steps=0 is rejected by the shared training config at submission.
                if data.get('max_steps')==0:
                    start.side_effect=ValueError('Invalid max_steps')
                    self.assertEqual(self.client.post('/api/training',json=data).status_code,400)
                else:
                    self.assertEqual(self.client.post('/api/training',json=data).status_code,422)
                    start.assert_not_called()
    def test_async_submission_returns_handle(self):
        with patch('app.main.training_ui.start',return_value={'config':{'run_id':'ui-test'},'status':'queued'}) as start:
            response=self.client.post('/api/training',json={'max_steps':2})
            self.assertEqual(response.status_code,202)
            self.assertEqual(start.call_args.args[0]['max_steps'],2)
    def test_download_paths_are_allowlisted_and_authenticated(self):
        self.assertEqual(self.client.get('/api/training/test/download?file=../../.env').status_code,422)
        app.dependency_overrides.clear()
        self.assertIn(self.client.get('/api/training').status_code,(401,503))
    def test_pending_job_has_no_metrics_yet(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'ui-test.json').write_text(json.dumps({'config':{'run_id':'ui-test'},'status':'queued'}))
            with patch.object(training_ui,'JOBS',Path(directory)),patch.object(training_ui,'read_bytes',side_effect=FileNotFoundError):
                self.assertEqual(training_ui.status('ui-test')['status'],'queued')
    def test_failed_call_overrides_stale_running_metadata(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'ui-test.json').write_text(json.dumps({'call_id':'fc-test'}))
            with patch.object(training_ui,'JOBS',Path(directory)),patch('modal.FunctionCall.from_id') as call,patch.object(training_ui,'read_bytes') as read:
                call.return_value.get.side_effect=RuntimeError('secret')
                read.side_effect=[json.dumps({'config':{'run_id':'ui-test'},'status':'running'}).encode(),b'']
                result=training_ui.status('ui-test')
                self.assertEqual(result['status'],'failed')
                self.assertNotIn('secret',str(result))
    def test_unfinished_training_cannot_publish(self):
        with patch.object(training_ui,'status',return_value={'status':'running'}):
            with self.assertRaises(ValueError): training_ui.publish(MagicMock(),'ui-test')
