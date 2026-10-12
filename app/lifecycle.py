"""Foreground supervisor for the local website and its dedicated cloud resources.

No PID-based killing: a locked instance consumes a token-scoped stop request.
Cloud shutdown preserves Neon data and Modal named Volumes/Secrets.
"""
import argparse
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import uvicorn

from app.config import Settings
from app.db import Repository

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "data" / "runtime"
MODAL_APP = "localize-llm-qwen3"


class LifecycleError(Exception):
    """Messages in this exception are safe to display without credentials."""


def failure(exc):
    return str(exc) if isinstance(exc, LifecycleError) else type(exc).__name__


class CloudLifecycle:
    def __init__(self, settings, cancelled=None):
        self.settings = settings
        self.cancelled = cancelled or threading.Event()

    def validate_config(self):
        names = ("database_url", "app_api_key", "vllm_base_url", "vllm_api_key",
                 "neon_api_key", "neon_project_id", "neon_endpoint_id",
                 "modal_profile", "modal_workspace", "modal_environment")
        missing = [n.upper() for n in names if not (
            getattr(self.settings, n).get_secret_value() if n.endswith("key") or n == "database_url"
            else getattr(self.settings, n))]
        if missing:
            raise LifecycleError("缺少設定：" + ", ".join(missing))
        target = urlparse(self.settings.vllm_base_url)
        expected = f"{self.settings.modal_workspace}--{MODAL_APP}-serve.modal.run"
        if (target.scheme != "https" or target.hostname != expected or target.path.rstrip("/") != "/v1"
                or target.username or target.password or target.query or target.fragment):
            raise LifecycleError("VLLM_BASE_URL 必須對應指定 Modal workspace 的 localize-llm-qwen3 端點")
        if not importlib.util.find_spec("modal"):
            raise LifecycleError("請先安裝 requirements-modal.txt")

    def neon(self, method="GET", suffix=""):
        s = self.settings
        path = f"https://console.neon.tech/api/v2/projects/{s.neon_project_id}/endpoints/{s.neon_endpoint_id}{suffix}"
        with httpx.Client(timeout=20) as client:
            response = client.request(method, path, headers={
                "Authorization": "Bearer " + s.neon_api_key.get_secret_value()})
        if response.status_code >= 400:
            raise LifecycleError(f"Neon 管理 API 回傳 HTTP {response.status_code}")
        return response.json()

    def verify_neon_target(self):
        endpoint = self.neon()["endpoint"]
        host = urlparse(self.settings.database_url.get_secret_value()).hostname or ""
        labels = host.split(".")
        labels[0] = labels[0].removesuffix("-pooler")
        if (endpoint.get("id") != self.settings.neon_endpoint_id
                or endpoint.get("host") != ".".join(labels)):
            raise LifecycleError("Neon 管理端點與 DATABASE_URL 不一致，未執行雲端變更")
        return endpoint

    def modal(self, arguments, timeout=90, cancellable=True):
        command = [sys.executable, "-m", "modal", *arguments]
        if arguments[0] != "profile":
            command += ["--env", self.settings.modal_environment, "--profile", self.settings.modal_profile]
        environment = dict(os.environ)
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN"):
            environment.pop(key, None)
        # Capture output without echoing CLI logs that may contain configuration.
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancellable and self.cancelled.is_set():
                        raise LifecycleError("啟動已取消")
                    if time.monotonic() >= deadline:
                        raise LifecycleError("Modal 指令逾時；遠端狀態需確認")
                    time.sleep(.2)
                if process.returncode:
                    raise LifecycleError(f"Modal 指令失敗（exit {process.returncode}）；請檢查指定 profile 的登入與部署設定")
                output.seek(0)
                return output.read().decode()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def modal_apps(self, cancellable=True):
        rows = json.loads(self.modal(["app", "list", "--json"], cancellable=cancellable))
        if not isinstance(rows, list):
            raise LifecycleError("Modal 清單格式不正確")
        return [r for r in rows if r.get("Description") == MODAL_APP]

    def verify_modal_profile(self, cancellable=True):
        profiles = json.loads(self.modal(["profile", "list", "--json"], cancellable=cancellable))
        if not any(p.get("name") == self.settings.modal_profile
                   and p.get("workspace") == self.settings.modal_workspace for p in profiles):
            raise LifecycleError("Modal profile 與指定 workspace 不一致或認證失敗")

    def preflight(self):
        self.validate_config()
        self.verify_neon_target()
        self.verify_modal_profile()
        # Check credentials before either cloud service is started.
        self.modal_apps()

    def start(self):
        print("連線 Neon，核對工作區與預設模型…", flush=True)
        repo = Repository(self.settings)
        repo.validate()
        model = repo.active_model()
        if not model:
            raise LifecycleError("Neon 尚未設定共用預設模型")
        if self.cancelled.is_set():
            raise LifecycleError("啟動已取消")
        print("部署 Modal 推論服務…", flush=True)
        self.modal(["deploy", "deploy/modal_vllm.py"], timeout=self.settings.project_start_timeout)
        print("等待 Modal 模型可用（冷啟動可能需數分鐘）…", flush=True)
        deadline = time.monotonic() + self.settings.project_start_timeout
        while time.monotonic() < deadline:
            if self.cancelled.is_set():
                raise LifecycleError("啟動已取消")
            try:
                with httpx.Client(timeout=20) as client:
                    response = client.get(self.settings.vllm_base_url.rstrip("/") + "/models",
                        headers={"Authorization": "Bearer " + self.settings.vllm_api_key.get_secret_value()})
                if response.status_code in (401, 403):
                    raise LifecycleError("Modal 推論金鑰無效")
                if response.status_code == 200:
                    served = {m["id"] for m in response.json()["data"]}
                    if model["served_name"] not in served:
                        raise LifecycleError("預設模型尚未載入 Modal，請確認模型登錄／LoRA 部署")
                    print("Neon 與 Modal 已確認可用。", flush=True)
                    return
            except httpx.TransportError:
                pass
            self.cancelled.wait(2)
        raise LifecycleError("Modal 模型就緒檢查逾時")

    def stop_modal(self):
        self.verify_modal_profile(cancellable=False)
        rows = self.modal_apps(cancellable=False)
        active = [r for r in rows if str(r.get("State", "")).lower() != "stopped"]
        for row in active:
            # Use exact IDs to avoid stopping a different deployment with the same name mid-command.
            self.modal(["app", "stop", row["App ID"], "--yes"], cancellable=False)
        if any(str(r.get("State", "")).lower() != "stopped" or int(r.get("Tasks", 0))
               for r in self.modal_apps(cancellable=False)):
            raise LifecycleError("Modal 尚未確認停止，請再次執行 stop")

    def stop_neon(self):
        endpoint = self.verify_neon_target()
        if endpoint.get("current_state") != "idle":
            self.neon("POST", "/suspend")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.neon()["endpoint"].get("current_state") == "idle":
                return
            time.sleep(1)
        raise LifecycleError("Neon 尚未確認暫停，其他用戶端可能仍在連線")

    def stop(self):
        results = {}
        for name, action in (("modal", self.stop_modal), ("neon", self.stop_neon)):
            try:
                action()
                results[name] = {"ok": True}
                print(f"{name}: 已確認停止運算", flush=True)
            except Exception as exc:
                results[name] = {"ok": False, "error": failure(exc)}
                print(f"{name}: 未確認停止（{failure(exc)}）", flush=True)
        return results


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def acquire_lock(stream):
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def run_project(settings, runtime, cloud=None, server_factory=uvicorn.Server):
    cancelled = threading.Event()
    cloud = cloud or CloudLifecycle(settings, cancelled)
    token = uuid4().hex
    state = {"token": token, "phase": "starting", "port": settings.project_port}
    state_path, stop_path = runtime / "state.json", runtime / "stop.json"
    server, watcher = None, None
    touched = False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    previous = {}

    def request_stop(*_):
        cancelled.set()
        if server is not None:
            server.should_exit = True

    def watch():
        while not cancelled.wait(.2):
            if read_state(stop_path).get("token") == token:
                request_stop()
            if server is not None and server.started and state["phase"] == "starting":
                state["phase"] = "ready"
                write_state(state_path, state)

    try:
        write_state(state_path, state)
        # Bind first so a conflicting Docker/service never causes cloud mutations.
        try:
            sock.bind(("127.0.0.1", settings.project_port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise LifecycleError(f"本機 {settings.project_port} 埠已被占用，請設定其他 PROJECT_PORT；未關閉占用程序") from None
            raise
        cloud.validate_config()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        cloud.preflight()
        if cancelled.is_set():
            raise LifecycleError("啟動已取消")
        touched = True
        cloud.start()
        if cancelled.is_set():
            raise LifecycleError("啟動已取消")
        server = server_factory(uvicorn.Config("app.main:app", host="127.0.0.1", port=settings.project_port,
                               timeout_graceful_shutdown=20, access_log=False))
        print(f"開啟 http://127.0.0.1:{settings.project_port}/documents；Ctrl+C 或 ./project stop 一起關閉。", flush=True)
        server.run(sockets=[sock])
        if not server.started:
            raise LifecycleError("本機網站啟動失敗")
    except Exception as exc:
        state["error"] = failure(exc)
        print("啟動／執行失敗：" + failure(exc), flush=True)
    finally:
        request_stop()
        if watcher:
            watcher.join(timeout=2)
        sock.close()
        state["phase"] = "stopping"
        write_state(state_path, state)
        print("本次管理的本機網站已結束；未關閉其他程序。", flush=True)
        state["shutdown"] = cloud.stop() if touched else {}
        success = touched and all(r["ok"] for r in state["shutdown"].values())
        state["phase"] = "stopped" if success else "failed"
        write_state(state_path, state)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if success and "error" not in state else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="啟動／關閉本機網站、Neon compute 與 Modal 推論部署（macOS/Linux）")
    parser.add_argument("command", choices=("start", "stop", "status"))
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (RUNTIME / "lock").open("a+") as lock:
        owned = acquire_lock(lock)
        state = read_state(RUNTIME / "state.json")
        if args.command == "status":
            print(json.dumps({"managed_process_running": not owned, "last_state": state}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "stop" and not owned:
            # A new supervisor may hold the lock before replacing its predecessor's state.
            initializing_until = time.monotonic() + 5
            while state.get("phase") not in ("starting", "ready", "stopping") and time.monotonic() < initializing_until:
                if acquire_lock(lock):
                    state = read_state(RUNTIME / "state.json")
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    return 0 if state.get("phase") == "stopped" else 1
                time.sleep(.05)
                state = read_state(RUNTIME / "state.json")
            if not state.get("token"):
                print("啟動程序正在初始化，請稍後再試。")
                return 1
            write_state(RUNTIME / "stop.json", {"token": state["token"]})
            print("已要求關閉，等待網站結束請求與雲端停止…", flush=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if acquire_lock(lock):
                    state = read_state(RUNTIME / "state.json")
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    return 0 if state.get("phase") == "stopped" else 1
                time.sleep(.2)
            print("關閉等待逾時；請用 ./project status 查看狀態。")
            return 1
        if not owned:
            print("專案已在啟動或執行中，請用 ./project status 查看狀態。")
            return 1
        try:
            settings = Settings()
            if args.command == "start":
                return run_project(settings, RUNTIME)
            # Recovery after a crashed supervisor; never kill unrelated local processes.
            cloud = CloudLifecycle(settings)
            cloud.validate_config()
            results = cloud.stop()
            ok = all(r["ok"] for r in results.values())
            write_state(RUNTIME / "state.json", {"phase": "stopped" if ok else "failed", "shutdown": results})
            print("沒有由此指令管理的本機網站；未關閉其他程序。")
            return 0 if ok else 1
        except Exception as exc:
            print(failure(exc))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
