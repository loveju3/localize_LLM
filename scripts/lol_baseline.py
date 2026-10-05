"""Run the versioned LoL test set through the local HTTP API (real GPU calls)."""
import argparse
import json
from pathlib import Path

import httpx
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = dotenv_values(".env")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.output.open("x", encoding="utf-8") as report, httpx.Client(
        base_url="http://127.0.0.1:8000", timeout=650,
        headers={"Authorization": "Bearer " + config["APP_API_KEY"]},
    ) as client:
        manifest = json.loads(Path('datasets/lol/v1/manifest.json').read_text())
        report.write(json.dumps({'type':'header', 'dataset':manifest, 'model_id':args.model}) + '\n')
        page = 1
        while True:
            response = client.get("/api/lol/cases", params={"split": "test", "page": page, "page_size": 20})
            response.raise_for_status()
            cases = response.json()["cases"]
            if not cases:
                break
            for case in cases:
                response = client.post("/api/lol/run", json={"case_id": case["id"], "model_id": args.model})
                if response.is_success:
                    result = {"status": "ok", "suite": case["suite"], **response.json()}
                else:
                    failures += 1
                    result = {"status": "error", "case_id": case["id"], "http_status": response.status_code}
                report.write(json.dumps(result, ensure_ascii=False) + "\n")
                report.flush()
                print(case["id"], result["status"], flush=True)
            page += 1
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
