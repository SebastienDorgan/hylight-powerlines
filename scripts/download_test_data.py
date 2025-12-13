import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

try:
    GDriveFileSystem = cast(
        Any,
        importlib.import_module("pydrive2.fs").GDriveFileSystem,
    )
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing optional dependency 'pydrive2'.\n"
        "This repo no longer pins pydrive2 because its current releases constrain "
        "'cryptography<44' (flagged by OSV).\n"
        "If you still want to use this script, install pydrive2 in a separate environment "
        "or add it locally, then rerun."
    ) from e

HYLIGHT_FOLDER_ID = os.getenv("HYLIGHT_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
NEW_ROOT = "assets/images"

if HYLIGHT_FOLDER_ID is None or GOOGLE_SERVICE_ACCOUNT_JSON is None:
    print(
        "Please set HYLIGHT_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_JSON environement variables first."
    )
    sys.exit(1)

fs = GDriveFileSystem(
    HYLIGHT_FOLDER_ID,
    use_service_account=True,
    client_json_file_path=GOOGLE_SERVICE_ACCOUNT_JSON,
)


def normalize_name(full_name: str) -> str:
    return full_name[-12:]


Path(NEW_ROOT).mkdir(parents=True, exist_ok=True)

# Sanity check
print("fs.root =", fs.root)

# 1) Simple non-recursive glob: files directly in that folder
for file in fs.glob(f"{fs.root}/*.JPG"):
    new_name = f"{NEW_ROOT}/{normalize_name(file)}"
    print(f"{fs.root}{normalize_name(file)}")
    fs.get(file, new_name)
