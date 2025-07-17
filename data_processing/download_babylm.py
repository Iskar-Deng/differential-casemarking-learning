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
        repo_type="dataset"
    )

    final_path = os.path.join(raw_dir, "cbt.txt")
    shutil.copy(file_path, final_path)

    print(f"BabyLM cbt.txt copied to: {final_path}")

if __name__ == "__main__":
    download_cbttxt()
