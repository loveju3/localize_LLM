"""Save/select endpoint profiles without printing credentials."""
import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values, set_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["runpod", "modal"])
    parser.add_argument("--save-current", action="store_true")
    args = parser.parse_args()
    profile = Path(f".env.{args.backend}")
    if args.save_current:
        values = dotenv_values(".env")
    else:
        values = dotenv_values(profile)
    url, key = values.get("VLLM_BASE_URL"), values.get("VLLM_API_KEY")
    if not url or not key or urlparse(url).scheme != "https" or not url.rstrip("/").endswith("/v1"):
        parser.error("Profile requires HTTPS VLLM_BASE_URL ending in /v1 and VLLM_API_KEY")
    destination = profile if args.save_current else Path(".env")
    if not destination.exists():
        descriptor = os.open(destination, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
    destination.chmod(0o600)
    set_key(destination, "VLLM_BASE_URL", url)
    set_key(destination, "VLLM_API_KEY", key)
    if not args.save_current:
        set_key(destination, "LLM_TIMEOUT_SECONDS", "600" if args.backend == "modal" else "180")
    print(f"{args.backend} profile {'saved' if args.save_current else 'selected'}; restart Uvicorn to apply.")


if __name__ == "__main__":
    main()
