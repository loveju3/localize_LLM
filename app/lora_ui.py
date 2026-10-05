"""Authenticated adapter upload and Modal staging; no client-supplied GPU paths."""
import tempfile
import json
from pathlib import Path
from app.artifacts import upload_lora, validate_identity

MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'

def save_upload(service, model_id, config_file, weights_file):
    validate_identity(model_id, MODEL, REVISION)
    with tempfile.TemporaryDirectory() as folder:
        for upload, name, limit in [(config_file, 'adapter_config.json', 65536),
                                    (weights_file, 'adapter_model.safetensors', 2 * 1024**3)]:
            if upload.filename != name:
                raise ValueError('請選擇 ' + name)
            path = Path(folder) / name
            size = 0
            with path.open('wb') as output:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError('LoRA 檔案超過大小限制')
                    output.write(chunk)
        config = json.loads((Path(folder) / 'adapter_config.json').read_text())
        if config.get('revision') != REVISION:
            raise ValueError('設定檔須包含與目前基礎模型一致的完整 revision')
        if type(config.get('r')) is not int or not 1 <= config['r'] <= 64:
            raise ValueError('頁面載入支援 rank 1～64')
        from safetensors import SafetensorError
        try:
            return upload_lora(service.repo, service.store, service.settings, folder,
                               model_id, MODEL, REVISION)
        except SafetensorError as exc:
            raise ValueError('權重檔不是有效的 safetensors') from exc

def load_adapter(service, model_id):
    model = service.repo.model(model_id)
    if not model or model['kind'] != 'lora':
        raise ValueError('請選擇已上傳的 LoRA')
    if (model['base_model'], model['base_revision']) != (MODEL, REVISION):
        raise ValueError('LoRA 與目前雲端基礎模型版本不相容')
    if '.modal.run' not in service.settings.vllm_base_url:
        raise ValueError('此頁面自動載入目前支援 Modal；RunPod 請使用 prepare-model 部署')
    if model['manifest'].get('rank', 999) > 64:
        raise ValueError('目前 Modal 部署支援 rank 上限 64')
    if model['served_name'] not in service.inference.served_models():
        import modal
        stage = modal.Function.from_name('localize-llm-qwen3', 'stage_adapter', environment_name='main')
        urls = {name: service.store.download_url(model['artifact_prefix'] + '/' + name)
                for name in model['manifest']['files']}
        stage.remote(model['id'], model['manifest'], urls)
        with service.inference.client() as client:
            result = client.post('load_lora_adapter', json={'lora_name': model['id'],
                                                          'lora_path': '/adapters/' + model['id']})
            result.raise_for_status()
    if model['served_name'] not in service.inference.served_models():
        raise ValueError('雲端尚未確認 LoRA 可用，請重新載入')
    return {'id': model['id'], 'status': 'available'}
