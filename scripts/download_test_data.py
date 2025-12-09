import os
import sys

from pydrive2.fs import GDriveFileSystem

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


os.makedirs(NEW_ROOT, exist_ok=True)

# Sanity check
print("fs.root =", fs.root)

# 1) Simple non-recursive glob: files directly in that folder
for file in fs.glob(f"{fs.root}/*.JPG"):
    new_name = f"{NEW_ROOT}/{normalize_name(file)}"
    print(f"{fs.root}{normalize_name(file)}")
    fs.get(file, new_name)
