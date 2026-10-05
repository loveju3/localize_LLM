"""Durable Modal training handles and allowlisted artifacts for the workspace UI."""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from app.artifacts import upload_lora
from training.lora import TrainConfig

JOBS = Path('data/training/jobs')

def volume():
    import modal
    return modal.Volume.from_name('localize-llm-training-runs', environment_name='main')

def read_bytes(run_id, relative):
    TrainConfig(run_id=run_id).validate()
    return b''.join(volume().read_file('/' + run_id + '/' + relative))

def start(parameters):
    import modal
    run_id = 'ui-' + uuid4().hex[:20]
    config = TrainConfig(run_id=run_id, **parameters)
    config.validate()
    JOBS.mkdir(parents=True, exist_ok=True)
    job = {'config': asdict(config), 'status': 'submitting'}
    path = JOBS / (run_id + '.json')
    path.write_text(json.dumps(job))
    try:
        fn = modal.Function.from_name('localize-llm-lora-training', 'train_job', environment_name='main')
        call = fn.spawn(asdict(config))
        job.update(status='queued', call_id=call.object_id)
    except Exception:
        job.update(status='failed', error_type='SubmissionError')
        raise ValueError('無法提交訓練，請確認 Modal 登入與訓練 App 部署')
    finally:
        path.write_text(json.dumps(job))
    return job

def status(run_id):
    import modal
    TrainConfig(run_id=run_id).validate()
    path = JOBS / (run_id + '.json')
    job = json.loads(path.read_text()) if path.exists() else {}
    terminal = None
    if job.get('call_id'):
        try:
            terminal = modal.FunctionCall.from_id(job['call_id']).get(timeout=0)
        except TimeoutError:
            pass
        except Exception as exc:
            terminal = {'status': 'failed', 'error_type': type(exc).__name__}
    try:
        result = json.loads(read_bytes(run_id, 'run.json'))
    except FileNotFoundError:
        if not job:
            raise LookupError('訓練不存在')
        result = job
    if terminal and (terminal.get('status') != 'failed' or result.get('status') != 'completed'):
        result = {**result, **terminal}
    try:
        metrics = [json.loads(line) for line in read_bytes(run_id, 'metrics.jsonl').decode().splitlines() if line.strip()]
    except FileNotFoundError:
        metrics = []
    return {**result, 'metrics': metrics}

def list_runs():
    names = {p.stem for p in JOBS.glob('*.json')} if JOBS.exists() else set()
    names.update(entry.path.strip('/') for entry in volume().iterdir('/', recursive=False)
                 if '/' not in entry.path.strip('/') and not entry.path.startswith('.'))
    rows = []
    for name in sorted(names, reverse=True)[:100]:
        try:
            row = status(name)
            rows.append({k:v for k,v in row.items() if k != 'metrics'})
        except (ValueError, LookupError):
            continue
    return rows

def publish(service, run_id):
    row = status(run_id)
    if row.get('status') != 'completed':
        raise ValueError('訓練完成後才能加入測試')
    config = row['config']
    existing = service.repo.model(run_id)
    if existing:
        if existing['kind'] != 'lora' or existing['manifest']['files']['adapter_model.safetensors'] != row['adapter_files']['adapter_model.safetensors']:
            raise ValueError('模型 ID 已被其他產物使用')
        return {'id': run_id}
    with tempfile.TemporaryDirectory() as folder:
        for name in ['adapter_config.json', 'adapter_model.safetensors']:
            data = read_bytes(run_id, 'adapter/' + name)
            import hashlib
            if hashlib.sha256(data).hexdigest() != row['adapter_files'][name]:
                raise ValueError('訓練產物雜湊不匹配')
            (Path(folder) / name).write_bytes(data)
        return upload_lora(service.repo, service.store, service.settings, folder,
                           run_id, config['model'], config['revision'])
