#!/usr/bin/env python3
"""Serve the sample-10 audit page and save edits back to the CSV.

Run from the repo root:
    python3 manual_audit/review_server.py
Then open:
    http://127.0.0.1:8765/sample_10_review.html
"""

from __future__ import annotations

import csv
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    from .audit_schema import AUDIT_COLUMNS
except ImportError:  # direct script execution
    from audit_schema import AUDIT_COLUMNS


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "sample_10_manual_audit.csv"
ROOT_CSV_PATH = HERE.parent / "sample_10_manual_audit.csv"
HTML_PATH = HERE / "sample_10_review.html"
ILLUSTRATION_DIR = HERE.parent / "newcomb_wildflower_guide" / "illustrations"
HOST = "127.0.0.1"
PORT = 8765

EXPECTED_COLUMNS = AUDIT_COLUMNS


def validate_csv_text(text: str) -> tuple[bool, str]:
    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error as error:
        return False, f"CSV parse error: {error}"
    if not rows:
        return False, "CSV is empty"
    if rows[0] != EXPECTED_COLUMNS:
        return False, "CSV header does not match the audit schema"
    if len(rows) < 2:
        return False, "CSV has no audit rows"
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(EXPECTED_COLUMNS):
            return False, f"Row {row_number} has {len(row)} columns"
    return True, "ok"


def write_csv_atomically(text: str) -> None:
    for destination in (CSV_PATH, ROOT_CSV_PATH):
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, destination)


class AuditHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self) -> None:
        if self.path in {"/", ""}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", f"/{HTML_PATH.name}")
            self.end_headers()
            return
        if self.path == "/api/load":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.end_headers()
            self.wfile.write(CSV_PATH.read_bytes())
            return
        illustration_prefix = "/newcomb_wildflower_guide/illustrations/"
        if self.path.startswith(illustration_prefix):
            filename = Path(self.path.removeprefix(illustration_prefix)).name
            asset = ILLUSTRATION_DIR / filename
            if not filename or not asset.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = asset.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON"})
            return

        csv_text = payload.get("csv")
        if not isinstance(csv_text, str):
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing CSV text"})
            return

        ok, message = validate_csv_text(csv_text)
        if not ok:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": message})
            return

        write_csv_atomically(csv_text)
        self.send_json(HTTPStatus.OK, {"ok": True})

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AuditHandler)
    url = f"http://{HOST}:{PORT}/{HTML_PATH.name}"
    print(f"Serving manual audit page at {url}")
    print(f"Edits will save to {CSV_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
