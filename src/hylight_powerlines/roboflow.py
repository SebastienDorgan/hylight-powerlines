"""Roboflow export downloader (ZIP-only), with retry logic for stale export links."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import httpx

LOG = logging.getLogger(__name__)
DEFAULT_API_BASE: Final[str] = "https://api.roboflow.com"
DEFAULT_API_KEY_ENV: Final[str] = "ROBOFLOW_API_KEY"

# Restrict to the formats you actually intend to use in this project.
ExportFormat = Literal[
    "yolov11",
    "yolov8",
]

ALLOWED_EXPORT_FORMATS: Final[set[str]] = {
    "yolov11",
    "yolov8",
}


@dataclass(slots=True)
class RoboflowDownloader:
    """Download datasets from Roboflow export endpoints.

    Attributes:
        workspace: Roboflow workspace name (for example, "poles-4ixf6").
        project: Roboflow project slug (for example, "power-pole-chyhd").
        version: Export version number, or None to automatically use the latest
            available version for the project.
        export_format: Export format string, restricted to known formats.
        api_base: Base URL for the Roboflow API.
        api_key_env: Name of the environment variable holding the API key.
    """

    workspace: str
    project: str
    version: int | None = None
    export_format: ExportFormat = "yolov8"
    api_base: str = DEFAULT_API_BASE
    api_key_env: str = DEFAULT_API_KEY_ENV
    client_factory: Callable[..., httpx.Client] = httpx.Client

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If the export format is not in the allowed set.
        """
        if self.export_format not in ALLOWED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export_format: {self.export_format!r}. "
                f"Allowed: {sorted(ALLOWED_EXPORT_FORMATS)}"
            )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def get_api_key(self) -> str:
        """Return the Roboflow API key from the configured environment variable.

        Returns:
            The API key string.

        Raises:
            RuntimeError: If the environment variable is not set.
        """
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} environment variable is not set")
        return api_key

    def _fetch_project_metadata(self, api_key: str | None = None) -> dict[str, Any]:
        """Fetch project-level metadata from Roboflow.

        This endpoint does not require a version and is used to discover the
        latest version number when `version` is None.

        Shape is typically:
            { "project": { "versions": <int>, ... }, ... }

        Args:
            api_key: Optional API key. If omitted, it is read from env.

        Returns:
            Parsed JSON dictionary.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
        """
        if api_key is None:
            api_key = self.get_api_key()

        url = f"{self.api_base}/{self.workspace}/{self.project}?api_key={api_key}"
        with self.client_factory(timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def _ensure_version(self, api_key: str | None = None) -> int:
        """Ensure that `self.version` is set; if None, resolve latest version.

        Returns:
            The resolved version number.

        Raises:
            RuntimeError: If the latest version cannot be determined.
        """
        if self.version is not None:
            return self.version

        metadata = self._fetch_project_metadata(api_key=api_key)
        # Try to be robust to different shapes
        project = metadata.get("project", metadata)
        versions = project.get("versions")

        # Common case: "versions" is an integer count; assume versions 1..N
        if isinstance(versions, int) and versions >= 1:
            latest = versions
        else:
            raise RuntimeError(
                f"Could not determine latest version from project metadata: {metadata!r}"
            )

        self.version = latest
        LOG.info(f"[RoboflowDownloader] Resolved latest version: {latest}")
        return latest

    def build_export_url(self, api_key: str | None = None) -> str:
        """Build the Roboflow export endpoint URL.

        Args:
            api_key: Optional API key. If omitted, the key is read from
                the environment using the configured environment variable.

        Returns:
            Fully qualified export URL.
        """
        if api_key is None:
            api_key = self.get_api_key()

        version = self._ensure_version(api_key=api_key)

        return (
            f"{self.api_base}/{self.workspace}/{self.project}/"
            f"{version}/{self.export_format}?api_key={api_key}"
        )

    def download_export_metadata(self, api_key: str | None = None) -> dict[str, Any]:
        """Call the Roboflow export endpoint and return the JSON metadata.

        The endpoint returns a JSON payload that contains an `export.link`
        URL from which you can download the ZIP file.

        Args:
            api_key: Optional API key. If omitted, the key is read from
                the environment.

        Returns:
            Parsed JSON dictionary returned by the export endpoint.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
        """
        url = self.build_export_url(api_key=api_key)
        with self.client_factory(timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def extract_export_link(metadata: dict[str, Any]) -> str:
        """Extract the ZIP download URL from Roboflow's export metadata.

        Args:
            metadata: JSON dictionary returned by the export endpoint.

        Returns:
            Direct URL from which to download the ZIP file.

        Raises:
            KeyError: If the expected keys are missing.
        """
        # Expected shape: { "export": { "link": "https://..." }, ... }
        return metadata["export"]["link"]

    def download_zip(self, zip_url: str, dest: Path) -> Path:
        """Download the dataset ZIP from `zip_url` to `dest`.

        Args:
            zip_url: Direct (or redirecting) URL to the ZIP file.
            dest: Path where the ZIP should be saved. The parent directory
                is created if it does not exist.

        Returns:
            The absolute path of the downloaded ZIP file.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        # follow_redirects=True is important for Roboflow's links.
        with (
            self.client_factory(timeout=None, follow_redirects=True) as client,
            client.stream("GET", zip_url) as resp,
        ):
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes():
                    if chunk:
                        f.write(chunk)

        return dest.resolve()

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def download_dataset(
        self,
        dest: Path,
        *,
        log_metadata: bool = True,
        metadata_truncate: int | None = None,
        max_retries: int = 1,
    ) -> Path:
        """Download a Roboflow dataset as a ZIP file.

        This method performs the full workflow:

        1. If `version` is None, resolve the latest version.
        2. Fetch export metadata for that version.
        3. Extract the ZIP URL.
        4. Stream the ZIP file to ``dest``.

        If the storage URL returns a 404 (stale export link), the export
        is re-triggered and the download is retried up to ``max_retries``
        additional times.

        Args:
            dest: Destination path for the ZIP file.
            log_metadata: If True, prints a truncated JSON dump of the metadata.
            metadata_truncate: Maximum number of characters of the JSON string
                to print. If None, the JSON is printed in full.
            max_retries: Number of extra attempts after the first try if a 404
                is encountered on the storage URL.

        Returns:
            The absolute path of the downloaded ZIP file.

        Raises:
            RuntimeError: If the API key is missing or the latest version
                cannot be determined.
            httpx.HTTPError: If any HTTP request fails and is not recoverable.
            KeyError: If the metadata does not contain an export link.
        """
        api_key = self.get_api_key()

        attempt = 0
        while True:
            metadata = self.download_export_metadata(api_key=api_key)

            if log_metadata:
                LOG.info("Export metadata:")
                text = json.dumps(metadata, indent=2)
                if metadata_truncate is not None and len(text) > metadata_truncate:
                    text = text[:metadata_truncate] + "\n... (truncated)"
                LOG.info(text)

            zip_url = self.extract_export_link(metadata)
            LOG.info(f"Downloading ZIP from: {zip_url}")

            try:
                out_zip = self.download_zip(zip_url, dest)
            except httpx.HTTPStatusError as exc:
                # Retry only on 404 from the storage backend
                if (
                    exc.response is not None
                    and exc.response.status_code == 404
                    and attempt < max_retries
                ):
                    attempt += 1
                    LOG.info(
                        f"Got 404 from storage, retrying export "
                        f"(attempt {attempt + 1}/{max_retries + 1})..."
                    )
                    continue
                # Any other error, or retries exhausted -> propagate
                raise
            else:
                LOG.info(f"Dataset ZIP saved to: {out_zip}")
                return out_zip
