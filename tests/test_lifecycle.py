import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from app.config import Settings
from app.lifecycle import CloudLifecycle, LifecycleError, acquire_lock, read_state, run_project, write_state


def config(**kwargs):
    values = dict(database_url='postgresql://user:secret@ep-demo-pooler.example.neon.tech/db',
        app_api_key='local-secret', vllm_api_key='model-secret',
        vllm_base_url='https://team--localize-llm-qwen3-serve.modal.run/v1',
        neon_api_key='neon-secret', neon_project_id='project-demo', neon_endpoint_id='ep-demo',
        modal_profile='profile-demo', modal_workspace='team')
    values.update(kwargs)
    return Settings(_env_file=None, **values)


class CloudLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.cloud = CloudLifecycle(config())

    def test_missing_management_settings_fail_without_network(self):
        cloud = CloudLifecycle(config(neon_api_key=''))
        with self.assertRaisesRegex(LifecycleError, 'NEON_API_KEY'):
            cloud.validate_config()

    def test_wrong_modal_endpoint_rejected(self):
        cloud = CloudLifecycle(config(vllm_base_url='https://other.modal.run/v1'))
        with self.assertRaisesRegex(LifecycleError, 'VLLM_BASE_URL'):
            cloud.validate_config()

    def test_neon_target_matches_pooled_database_hostname(self):
        with patch.object(self.cloud, 'neon', return_value={'endpoint': {
            'id': 'ep-demo', 'host': 'ep-demo.example.neon.tech'}}):
            self.cloud.verify_neon_target()

    def test_wrong_neon_target_prevents_suspend(self):
        with patch.object(self.cloud, 'neon', return_value={'endpoint': {
            'id': 'ep-other', 'host': 'ep-other.example.neon.tech'}}) as request:
            with self.assertRaises(LifecycleError):
                self.cloud.stop_neon()
            request.assert_called_once_with()

    def test_suspend_is_verified_without_sql_reawakening(self):
        replies = [{'endpoint': {'id': 'ep-demo', 'host': 'ep-demo.example.neon.tech', 'current_state': 'active'}},
                   {'operations': []}, {'endpoint': {'current_state': 'idle'}}]
        with patch.object(self.cloud, 'neon', side_effect=replies) as request:
            self.cloud.stop_neon()
        self.assertEqual(request.call_args_list[1].args, ('POST', '/suspend'))

    def test_already_idle_is_idempotent(self):
        endpoint = {'id': 'ep-demo', 'host': 'ep-demo.example.neon.tech', 'current_state': 'idle'}
        with patch.object(self.cloud, 'neon', return_value={'endpoint': endpoint}) as request:
            self.cloud.stop_neon()
        self.assertTrue(all(not c.args for c in request.call_args_list))

    def test_neon_management_http_and_redacted_errors(self):
        captured = []
        def handler(request):
            captured.append(request)
            return httpx.Response(403, text='neon-secret')
        original = httpx.Client
        with patch('app.lifecycle.httpx.Client', side_effect=lambda **kw: original(transport=httpx.MockTransport(handler), **kw)):
            with self.assertRaisesRegex(LifecycleError, 'HTTP 403') as error:
                self.cloud.neon('POST', '/suspend')
        self.assertNotIn('neon-secret', str(error.exception))
        self.assertEqual(str(captured[0].url), 'https://console.neon.tech/api/v2/projects/project-demo/endpoints/ep-demo/suspend')

    def test_modal_stop_uses_exact_id_and_checks_result(self):
        rows = [{'Description': 'localize-llm-qwen3', 'App ID': 'ap-1', 'State': 'deployed', 'Tasks': '1'}]
        with patch.object(self.cloud, 'verify_modal_profile'), \
             patch.object(self.cloud, 'modal_apps', side_effect=[rows, []]), \
             patch.object(self.cloud, 'modal') as command:
            self.cloud.stop_modal()
        command.assert_called_once_with(['app', 'stop', 'ap-1', '--yes'], cancellable=False)

    def test_wrong_modal_profile_prevents_stop(self):
        with patch.object(self.cloud, 'modal', return_value=json.dumps([
            {'name': 'profile-demo', 'workspace': 'wrong'}])) as command:
            with self.assertRaises(LifecycleError):
                self.cloud.stop_modal()
        self.assertEqual(command.call_count, 1)

    def test_shutdown_attempts_neon_after_modal_failure(self):
        with patch.object(self.cloud, 'stop_modal', side_effect=RuntimeError('secret')), \
             patch.object(self.cloud, 'stop_neon') as neon:
            result = self.cloud.stop()
        neon.assert_called_once()
        self.assertFalse(result['modal']['ok'])
        self.assertEqual(result['modal']['error'], 'RuntimeError')
        self.assertTrue(result['neon']['ok'])

    def test_start_requires_active_model_before_deploy(self):
        repo = MagicMock()
        repo.active_model.return_value = None
        with patch('app.lifecycle.Repository', return_value=repo), patch.object(self.cloud, 'modal') as deploy:
            with self.assertRaisesRegex(LifecycleError, '預設模型'):
                self.cloud.start()
        deploy.assert_not_called()

    def test_start_waits_for_registered_model(self):
        repo = MagicMock()
        repo.active_model.return_value = {'served_name': 'base'}
        replies = [httpx.Response(503), httpx.Response(200, json={'data': [{'id':'base'}]})]
        original = httpx.Client
        with patch('app.lifecycle.Repository', return_value=repo), patch.object(self.cloud, 'modal') as deploy, \
             patch.object(self.cloud.cancelled, 'wait'), \
             patch('app.lifecycle.httpx.Client', side_effect=lambda **kw: original(
                 transport=httpx.MockTransport(lambda request: replies.pop(0)), **kw)):
            self.cloud.start()
        repo.validate.assert_called_once()
        self.assertEqual(deploy.call_args.args[0], ['deploy', 'deploy/modal_vllm.py'])
        self.assertEqual(len(replies), 0)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runtime = Path(self.directory.name)
        self.cloud = MagicMock()
        self.cloud.stop.return_value = {'modal': {'ok': True}, 'neon': {'ok': True}}
        self.server = MagicMock()
        self.server.started = True

    def run_project(self):
        with patch('app.lifecycle.socket.socket'):
            return run_project(config(), self.runtime, self.cloud, lambda cfg: self.server)

    def test_start_before_website_and_cleanup_after_exit(self):
        actions = []
        self.cloud.start.side_effect = lambda: actions.append('cloud_ready')
        self.server.run.side_effect = lambda **kw: actions.append('web')
        self.cloud.stop.side_effect = lambda: actions.append('cloud_stop') or {'modal': {'ok': True}, 'neon': {'ok': True}}
        self.assertEqual(self.run_project(), 0)
        self.assertEqual(actions, ['cloud_ready', 'web', 'cloud_stop'])
        self.assertEqual(read_state(self.runtime / 'state.json')['phase'], 'stopped')

    def test_preflight_failure_does_not_start_or_stop_cloud(self):
        self.cloud.preflight.side_effect = LifecycleError('bad credentials')
        self.assertEqual(self.run_project(), 1)
        self.cloud.start.assert_not_called()
        self.cloud.stop.assert_not_called()
        self.server.run.assert_not_called()

    def test_start_failure_cleans_up_and_never_opens_website(self):
        self.cloud.start.side_effect = RuntimeError('private secret')
        self.assertEqual(self.run_project(), 1)
        self.cloud.stop.assert_called_once()
        self.server.run.assert_not_called()
        self.assertNotIn('private secret', (self.runtime / 'state.json').read_text())

    def test_shutdown_failure_remains_visible(self):
        self.cloud.stop.return_value = {'modal': {'ok': False}, 'neon': {'ok': True}}
        self.assertEqual(self.run_project(), 1)
        self.assertEqual(read_state(self.runtime / 'state.json')['phase'], 'failed')

    def test_occupied_port_does_not_touch_cloud(self):
        with patch('app.lifecycle.socket.socket') as sock:
            sock.return_value.bind.side_effect = OSError('address in use')
            self.assertEqual(run_project(config(), self.runtime, self.cloud), 1)
        self.cloud.preflight.assert_not_called()
        self.cloud.start.assert_not_called()
        self.cloud.stop.assert_not_called()

    def test_lock_prevents_duplicate_supervisor(self):
        with (self.runtime / 'lock').open('a+') as first, (self.runtime / 'lock').open('a+') as second:
            self.assertTrue(acquire_lock(first))
            self.assertFalse(acquire_lock(second))

    def test_stop_file_requests_graceful_exit(self):
        def serve(**kw):
            self.server.should_exit = False
            state = read_state(self.runtime / 'state.json')
            write_state(self.runtime / 'stop.json', {'token': state['token']})
            deadline = __import__('time').monotonic() + 3
            while not self.server.should_exit and __import__('time').monotonic() < deadline:
                threading.Event().wait(.05)
            self.assertTrue(self.server.should_exit)
        self.server.run.side_effect = serve
        self.assertEqual(self.run_project(), 0)

    def test_stale_stop_file_does_not_cancel_next_start(self):
        write_state(self.runtime / 'stop.json', {'token': 'old-instance'})
        self.assertEqual(self.run_project(), 0)
        self.server.run.assert_called_once()
