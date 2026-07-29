"""Start the Streamlit server and require a healthy local endpoint."""

from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "TSA_Data_Science_Analytics.py"


def _available_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def main() -> None:
    port = _available_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + 30
    output = ""
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(
                    "Streamlit exited before becoming healthy:\n" + output
                )
            try:
                with urlopen(health_url, timeout=1) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    if response.status == 200 and body.strip() == "ok":
                        print(f"streamlit_smoke_ok port={port}")
                        return
            except (URLError, TimeoutError):
                time.sleep(0.2)
        output = process.stdout.read() if process.poll() is not None else ""
        raise RuntimeError(
            f"Streamlit did not become healthy within 30 seconds:\n{output}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
