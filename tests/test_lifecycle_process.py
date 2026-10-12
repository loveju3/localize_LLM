"""Opt-in localhost integration: RUN_LOCAL_LIFECYCLE_TESTS=1, no cloud access."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(os.environ.get('RUN_LOCAL_LIFECYCLE_TESTS') == '1', 'Opt-in localhost socket test')
class LifecycleProcessTests(unittest.TestCase):
    def test_real_start_status_stop_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            runner = root / 'runner.py'
            runner.write_text('''
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import app.lifecycle as lifecycle
from app.config import Settings
lifecycle.RUNTIME = Path(sys.argv[2])
port = int(sys.argv[3])
class FakeCloud:
    def __init__(self, settings, cancelled=None): pass
    def validate_config(self): pass
    def preflight(self): pass
    def start(self):
        (lifecycle.RUNTIME / 'cloud-started').write_text('ready')
    def stop(self):
        (lifecycle.RUNTIME / 'cloud-stopped').write_text('stopped')
        return {'modal': {'ok': True}, 'neon': {'ok': True}}
lifecycle.CloudLifecycle = FakeCloud
lifecycle.Settings = lambda: Settings(_env_file=None, project_port=port)
raise SystemExit(lifecycle.main([sys.argv[4]]))
''')
            runtime = root / 'runtime'
            command = [sys.executable, str(runner), str(ROOT), str(runtime), str(port)]
            for cycle in range(2):
                with self.subTest(cycle=cycle), (root / f'process-{cycle}.log').open('w+') as log:
                    process = subprocess.Popen([*command, 'start'], stdout=log, stderr=log)
                    try:
                        deadline = time.monotonic() + 15
                        ready = False
                        with httpx.Client(timeout=.5, trust_env=False) as client:
                            while time.monotonic() < deadline and process.poll() is None:
                                try:
                                    response = client.get(f'http://127.0.0.1:{port}/health')
                                    if response.status_code == 200:
                                        ready = True
                                        break
                                except httpx.TransportError:
                                    pass
                                time.sleep(.05)
                            log.flush()
                            log.seek(0)
                            self.assertTrue(ready, 'Local server failed to start: ' + log.read())
                            self.assertTrue((runtime / 'cloud-started').exists())
                            status = subprocess.run([*command, 'status'], capture_output=True, text=True, timeout=5)
                            self.assertTrue(json.loads(status.stdout)['managed_process_running'])
                            duplicate = subprocess.run([*command, 'start'], capture_output=True, text=True, timeout=5)
                            self.assertEqual(duplicate.returncode, 1)
                            stopped = subprocess.run([*command, 'stop'], capture_output=True, text=True, timeout=10)
                            self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                            self.assertEqual(process.wait(timeout=5), 0)
                            self.assertTrue((runtime / 'cloud-stopped').exists())
                            self.assertEqual(json.loads((runtime / 'state.json').read_text())['phase'], 'stopped')
                            with self.assertRaises(httpx.TransportError):
                                client.get(f'http://127.0.0.1:{port}/health')
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
