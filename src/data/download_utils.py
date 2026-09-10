"""HTTP download with resume and progress logging."""

from __future__ import annotations

import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024  # 1 MB


def parse_json_line(line: str) -> dict | None:
    """Parse one JSON or Python-literal line (Amazon/Steam official dumps)."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None
    return obj if isinstance(obj, dict) else None


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def download_file(
    url: str,
    dest: Path,
    *,
    log_interval: float = 10.0,
) -> Path:
    """Download *url* to *dest*, resuming *.part* if present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    req = Request(url, headers={"User-Agent": "ur4rec-dataset-downloader/1.0"})
    if resume_pos:
        req.add_header("Range", f"bytes={resume_pos}-")

    try:
        with urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", resp.getcode())
            if resume_pos and status == 200:
                tmp.unlink(missing_ok=True)
                resume_pos = 0
                req = Request(url, headers={"User-Agent": "ur4rec-dataset-downloader/1.0"})
                resp.close()
                resp = urlopen(req, timeout=60)

            content_range = resp.headers.get("Content-Range")
            content_length = resp.headers.get("Content-Length")
            if content_range and "/" in content_range:
                total = int(content_range.rsplit("/", 1)[-1])
            elif content_length:
                total = int(content_length) + resume_pos
            else:
                total = None

            mode = "ab" if resume_pos else "wb"
            downloaded = resume_pos
            t0 = time.time()
            last_log = t0

            _log(
                f"[download] {dest.name}: "
                f"{'resume' if resume_pos else 'start'} "
                f"from {_fmt_bytes(resume_pos)}"
                + (f" / {_fmt_bytes(total)}" if total else "")
            )

            with open(tmp, mode) as out:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= log_interval:
                        elapsed = now - t0
                        speed = (downloaded - resume_pos) / elapsed if elapsed > 0 else 0
                        if total:
                            left = total - downloaded
                            eta = left / speed if speed > 0 else float("inf")
                            pct = 100.0 * downloaded / total
                            _log(
                                f"[download] {dest.name}: "
                                f"{_fmt_bytes(downloaded)} / {_fmt_bytes(total)} "
                                f"({pct:.1f}%) | left {_fmt_bytes(left)} | "
                                f"speed {_fmt_bytes(int(speed))}/s | ETA {_fmt_eta(eta)}"
                            )
                        else:
                            _log(
                                f"[download] {dest.name}: "
                                f"{_fmt_bytes(downloaded)} | "
                                f"speed {_fmt_bytes(int(speed))}/s"
                            )
                        last_log = now

            elapsed = time.time() - t0
            speed = (downloaded - resume_pos) / elapsed if elapsed > 0 else 0
            _log(
                f"[download] {dest.name}: done "
                f"{_fmt_bytes(downloaded)} in {_fmt_eta(elapsed)} "
                f"({_fmt_bytes(int(speed))}/s avg)"
            )
    except HTTPError as e:
        if tmp.exists() and tmp.stat().st_size == 0:
            tmp.unlink()
        raise RuntimeError(f"HTTP {e.code} for {url}") from e
    except Exception:
        _log(
            f"[download] {dest.name}: interrupted, "
            f"partial kept at {tmp} ({_fmt_bytes(tmp.stat().st_size if tmp.exists() else 0)})"
        )
        raise

    tmp.replace(dest)
    return dest
