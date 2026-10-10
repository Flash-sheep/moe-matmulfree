#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import signal
import socket
import ssl
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
    "HF_ENDPOINT",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
]


DNS_TARGETS = [
    "huggingface.co",
    "pypi.org",
    "raw.githubusercontent.com",
]


HTTP_TARGETS = [
    ("huggingface_home", "https://huggingface.co"),
    ("huggingface_finewebedu_metadata", "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu"),
    ("pypi_simple_datasets", "https://pypi.org/simple/datasets/"),
    ("github_raw_readme", "https://raw.githubusercontent.com/huggingface/datasets/main/README.md"),
]


@contextlib.contextmanager
def no_proxy_env() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    try:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def timeout_guard(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"operation timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def capture_proxy_env() -> Dict[str, str]:
    return {key: value for key, value in ((key, os.environ.get(key)) for key in PROXY_ENV_KEYS) if value}


def run_dns_check(hostname: str) -> Dict[str, object]:
    start = time.time()
    try:
        resolved = socket.gethostbyname(hostname)
        return {
            "name": f"dns_{hostname}",
            "target": hostname,
            "success": True,
            "resolved_ip": resolved,
            "elapsed_sec": round(time.time() - start, 6),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": f"dns_{hostname}",
            "target": hostname,
            "success": False,
            "resolved_ip": None,
            "elapsed_sec": round(time.time() - start, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }


def make_opener(use_env_proxy: bool):
    handlers = []
    if not use_env_proxy:
        handlers.append(urllib_request.ProxyHandler({}))
    handlers.append(urllib_request.HTTPSHandler(context=ssl.create_default_context()))
    handlers.append(urllib_request.HTTPHandler())
    return urllib_request.build_opener(*handlers)


def run_http_check(name: str, url: str, timeout: int, use_env_proxy: bool) -> Dict[str, object]:
    opener = make_opener(use_env_proxy=use_env_proxy)
    methods = ["HEAD", "GET"]
    last_error: Optional[str] = None
    for method in methods:
        start = time.time()
        try:
            req = urllib_request.Request(
                url,
                method=method,
                headers={
                    "User-Agent": "moe-matmulfree-network-check/1.0",
                    "Accept": "*/*",
                },
            )
            with opener.open(req, timeout=timeout) as resp:
                status_code = getattr(resp, "status", None)
                content_type = resp.headers.get("Content-Type")
                bytes_read = 0
                if method == "GET":
                    payload = resp.read(4096)
                    bytes_read = len(payload)
                return {
                    "name": name,
                    "url": url,
                    "method": method,
                    "success": True,
                    "status_code": status_code,
                    "content_type": content_type,
                    "bytes_read": bytes_read,
                    "elapsed_sec": round(time.time() - start, 6),
                    "error": None,
                }
        except urllib_error.HTTPError as exc:
            bytes_read = 0
            if method == "GET":
                try:
                    bytes_read = len(exc.read(4096))
                except Exception:  # noqa: BLE001
                    bytes_read = 0
            return {
                "name": name,
                "url": url,
                "method": method,
                "success": False,
                "status_code": exc.code,
                "content_type": getattr(exc, "headers", {}).get("Content-Type") if getattr(exc, "headers", None) else None,
                "bytes_read": bytes_read,
                "elapsed_sec": round(time.time() - start, 6),
                "error": f"HTTPError: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if method == "HEAD":
                continue
            return {
                "name": name,
                "url": url,
                "method": method,
                "success": False,
                "status_code": None,
                "content_type": None,
                "bytes_read": 0,
                "elapsed_sec": round(time.time() - start, 6),
                "error": last_error,
            }
    return {
        "name": name,
        "url": url,
        "method": "HEAD_or_GET",
        "success": False,
        "status_code": None,
        "content_type": None,
        "bytes_read": 0,
        "elapsed_sec": 0.0,
        "error": last_error or "unknown error",
    }


def _fineweb_streaming_worker(queue, use_no_proxy: bool) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        queue.put({
            "name": "finewebedu_streaming_one_sample",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "streaming": True,
            "success": False,
            "elapsed_sec": 0.0,
            "error": f"datasets package not usable: {type(exc).__name__}: {exc}",
            "sample_schema": None,
            "text_length": None,
        })
        return

    try:
        context = no_proxy_env() if use_no_proxy else contextlib.nullcontext()
        with context:
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                split="train",
                streaming=True,
            )
            first = next(iter(ds))
        text_value = first.get("text") if isinstance(first, dict) else None
        schema = sorted(list(first.keys())) if isinstance(first, dict) else None
        queue.put({
            "name": "finewebedu_streaming_one_sample",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "streaming": True,
            "success": True,
            "elapsed_sec": 0.0,
            "error": None,
            "sample_schema": schema,
            "text_length": None if text_value is None else len(text_value),
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({
            "name": "finewebedu_streaming_one_sample",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "streaming": True,
            "success": False,
            "elapsed_sec": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "sample_schema": None,
            "text_length": None,
        })


def run_fineweb_streaming_check(timeout: int, use_no_proxy: bool) -> Dict[str, object]:
    start = time.time()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_fineweb_streaming_worker, args=(queue, use_no_proxy), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "name": "finewebedu_streaming_one_sample",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "streaming": True,
            "success": False,
            "elapsed_sec": round(time.time() - start, 6),
            "error": f"TimeoutError: operation timed out after {timeout}s",
            "sample_schema": None,
            "text_length": None,
        }
    if not queue.empty():
        result = queue.get()
        result["elapsed_sec"] = round(time.time() - start, 6)
        return result
    return {
        "name": "finewebedu_streaming_one_sample",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "streaming": True,
        "success": False,
        "elapsed_sec": round(time.time() - start, 6),
        "error": f"RuntimeError: worker exited with code {proc.exitcode} without returning a result",
        "sample_schema": None,
        "text_length": None,
    }


def summarize(report: Dict[str, object]) -> Dict[str, object]:
    def _lookup(name: str) -> Optional[bool]:
        for item in report["checks"]:
            if item.get("name") == name:
                return bool(item.get("success"))
        return None

    direct_hf = _lookup("huggingface_home")
    direct_meta = _lookup("huggingface_finewebedu_metadata")
    direct_stream = _lookup("finewebedu_streaming_one_sample")
    direct_pypi = _lookup("pypi_simple_datasets")
    direct_github = _lookup("github_raw_readme")

    recommendation = "unclear_need_manual_check"
    if all(value is True for value in [direct_hf, direct_meta, direct_stream, direct_pypi, direct_github]):
        recommendation = "direct_download_ok"
    elif any(value is False for value in [direct_hf, direct_meta, direct_pypi, direct_github]):
        current_env_checks = report.get("current_env_checks") or []
        current_env_success = any(bool(item.get("success")) for item in current_env_checks)
        if current_env_success:
            recommendation = "proxy_required"

    return {
        "direct_huggingface_access": direct_hf,
        "direct_finewebedu_metadata_access": direct_meta,
        "direct_finewebedu_streaming_one_sample": direct_stream,
        "direct_pypi_access": direct_pypi,
        "direct_github_raw_access": direct_github,
        "recommendation": recommendation,
    }


def build_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Direct Network Access Check",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- proxy_env_removed_for_check: `{report['proxy_env_removed_for_check']}`",
        "",
        "## Proxy Environment Before",
    ]
    if report["proxy_env_before"]:
        for key, value in report["proxy_env_before"].items():
            lines.append(f"- `{key}={value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## No-Proxy Checks"])
    for item in report["checks"]:
        status = "ok" if item.get("success") else "fail"
        target = item.get("url", item.get("target", item.get("dataset", "")))
        lines.append(f"- `{item['name']}`: `{status}` target=`{target}` error=`{item.get('error')}`")
    current_env_checks = report.get("current_env_checks") or []
    if current_env_checks:
        lines.extend(["", "## Current-Env Comparison Checks"])
        for item in current_env_checks:
            status = "ok" if item.get("success") else "fail"
            target = item.get("url", item.get("target", item.get("dataset", "")))
            lines.append(f"- `{item['name']}`: `{status}` target=`{target}` error=`{item.get('error')}`")
    lines.extend(["", "## Summary"])
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check direct no-proxy network access for dataset sources.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proxy_before = capture_proxy_env()
    checks = []

    for hostname in DNS_TARGETS:
        checks.append(run_dns_check(hostname))

    with no_proxy_env():
        for name, url in HTTP_TARGETS:
            checks.append(run_http_check(name=name, url=url, timeout=args.timeout, use_env_proxy=False))
        hf_home_ok = any(item["name"] == "huggingface_home" and bool(item["success"]) for item in checks)
        hf_meta_ok = any(item["name"] == "huggingface_finewebedu_metadata" and bool(item["success"]) for item in checks)
        if hf_home_ok and hf_meta_ok:
            checks.append(run_fineweb_streaming_check(timeout=args.timeout, use_no_proxy=False))
        else:
            checks.append(
                {
                    "name": "finewebedu_streaming_one_sample",
                    "dataset": "HuggingFaceFW/fineweb-edu",
                    "streaming": True,
                    "success": False,
                    "elapsed_sec": 0.0,
                    "error": "skipped: direct HuggingFace access failed before streaming check",
                    "sample_schema": None,
                    "text_length": None,
                }
            )

    report: Dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "proxy_env_before": proxy_before,
        "proxy_env_removed_for_check": True,
        "checks": checks,
    }

    if proxy_before and any(not bool(item.get("success")) for item in checks if item["name"] in {
        "huggingface_home",
        "huggingface_finewebedu_metadata",
        "pypi_simple_datasets",
        "github_raw_readme",
    }):
        current_env_checks = []
        for name, url in HTTP_TARGETS:
            current_env_checks.append(run_http_check(name=name, url=url, timeout=args.timeout, use_env_proxy=True))
        report["current_env_checks"] = current_env_checks

    report["summary"] = summarize(report)
    (args.output_dir / "network_direct_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "network_direct_check_report.md").write_text(
        build_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
