from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from hylight_powerlines.roboflow import RoboflowDownloader


def test_roboflow_export_format_validation() -> None:
    with pytest.raises(ValueError):
        RoboflowDownloader(workspace="w", project="p", export_format="yolov5")  # type: ignore[arg-type]


def test_build_export_url_resolves_latest_version_without_network(tmp_path: Path) -> None:
    api_key = "KEY"
    state: dict[str, int] = {"exports": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://api.roboflow.com/w/p?api_key="):
            return httpx.Response(200, json={"project": {"versions": 3}})
        if url.startswith("https://api.roboflow.com/w/p/3/yolov8?api_key="):
            state["exports"] += 1
            return httpx.Response(200, json={"export": {"link": "https://storage/zip_ok"}})
        if url == "https://storage/zip_ok":
            return httpx.Response(200, content=b"zip-bytes")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    downloader = RoboflowDownloader(
        workspace="w",
        project="p",
        version=None,
        export_format="yolov8",
        client_factory=lambda **kw: httpx.Client(transport=transport, **kw),
    )

    assert downloader.build_export_url(api_key=api_key).endswith("/w/p/3/yolov8?api_key=KEY")

    out = downloader.download_dataset(tmp_path / "ds.zip", log_metadata=False)
    assert out.is_file()
    assert out.read_bytes() == b"zip-bytes"
    assert state["exports"] == 1


def test_download_dataset_retries_stale_export_link_404(tmp_path: Path) -> None:
    state: dict[str, int] = {"export_calls": 0, "zip1_calls": 0, "zip2_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://api.roboflow.com/w/p?api_key="):
            return httpx.Response(200, json={"project": {"versions": 1}})

        if url.startswith("https://api.roboflow.com/w/p/1/yolov8?api_key="):
            state["export_calls"] += 1
            link = "https://storage/zip1" if state["export_calls"] == 1 else "https://storage/zip2"
            return httpx.Response(200, json={"export": {"link": link}})

        if url == "https://storage/zip1":
            state["zip1_calls"] += 1
            return httpx.Response(404)

        if url == "https://storage/zip2":
            state["zip2_calls"] += 1
            return httpx.Response(200, content=b"ok")

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    downloader = RoboflowDownloader(
        workspace="w",
        project="p",
        version=None,
        export_format="yolov8",
        client_factory=lambda **kw: httpx.Client(transport=transport, **kw),
    )

    out = downloader.download_dataset(tmp_path / "ds.zip", log_metadata=False, max_retries=1)
    assert out.read_bytes() == b"ok"
    assert state["export_calls"] == 2
    assert state["zip1_calls"] == 1
    assert state["zip2_calls"] == 1
