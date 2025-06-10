# data_processing/download_babylm.py

from huggingface_hub import hf_hub_download
import os
import shutil
from utils import DATA_PATH

def download_cbttxt():
    raw_dir = os.path.join(DATA_PATH, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    file_path = hf_hub_download(
        repo_id="cambridge-climb/BabyLM",
        filename="clean/100M/cbt.txt",
        repo_type="dataset",
        local_dir=raw_dir,
        local_dir_use_symlinks=False
    )

    final_path = os.path.join(raw_dir, "cbt.txt")
    shutil.move(file_path, final_path)

    print(f"BabyLM cbt.txt downloaded to: {final_path}")

    nested_path = os.path.dirname(file_path)
    while nested_path != raw_dir and os.path.isdir(nested_path):
        try:
            os.rmdir(nested_path)
        except OSError:
            break
        nested_path = os.path.dirname(nested_path)

if __name__ == "__main__":
    download_cbttxt()
