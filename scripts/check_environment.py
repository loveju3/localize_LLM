"""Offline dependency smoke check; does not contact cloud services or load models."""
import importlib
import platform
import sys

MODULES = (
    "fastapi", "uvicorn", "python_multipart", "pydantic_settings", "httpx",
    "pypdf", "sentence_transformers", "psycopg", "pgvector", "boto3",
    "docx", "openpyxl",
)


def main():
    print(f"Python {platform.python_version()} / {platform.machine()}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}")
    if failures:
        return 1
    print("Local dependencies ready. Cloud connections and model inference not tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
