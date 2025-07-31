import os
from utils import CONFIG_PATH, DATA_PATH, CHECKPOINT_PATH, CACHE_PATH, MISTRAL_PATH

DATASET_TEMPLATE_FILE = os.path.join(CONFIG_PATH, "dataset_template.yaml")
MAIN_TEMPLATE_FILE = os.path.join(CONFIG_PATH, "main_template.yaml")

DATASET_INPUT_ROOT = os.path.join(DATA_PATH, "perturbed_model")

DATASET_OUTPUT_DIR = os.path.join(MISTRAL_PATH, "conf/user_datasets")
MAIN_OUTPUT_DIR = os.path.join(MISTRAL_PATH, "conf/user_main")

SPLIT_DIRS = {
    "train_without_invalid": "",
    "train_with_invalid": "_with_invalid",
}

os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

dataset_template = load_template(DATASET_TEMPLATE_FILE)
main_template = load_template(MAIN_TEMPLATE_FILE)

for subdir in os.listdir(DATASET_INPUT_ROOT):
    subpath = os.path.join(DATASET_INPUT_ROOT, subdir)
    if not os.path.isdir(subpath):
        continue

    found_any = False
    for split_name, id_suffix in SPLIT_DIRS.items():
        dataset_dir = os.path.join(subpath, split_name)
        if not os.path.isdir(dataset_dir):
            continue

        found_any = True
        dataset_dir_abs = os.path.abspath(dataset_dir)

        dataset_id = f"{subdir}{id_suffix}"

        dataset_yaml = (dataset_template
            .replace("__DATASET_ID__", dataset_id)
            .replace("__DATASET_NAME__", dataset_id)
            .replace("__DATASET_DIR__", dataset_dir_abs))

        main_yaml = (main_template
            .replace("__DATASET_ID__", dataset_id)
            .replace("CHECKPOINT_PATH", CHECKPOINT_PATH)
            .replace("CACHE_PATH", CACHE_PATH))

        dataset_out_path = os.path.join(DATASET_OUTPUT_DIR, f"{dataset_id}.yaml")
        main_out_path = os.path.join(MAIN_OUTPUT_DIR, f"{dataset_id}.yaml")

        with open(dataset_out_path, "w", encoding="utf-8") as f:
            f.write(dataset_yaml)
        with open(main_out_path, "w", encoding="utf-8") as f:
            f.write(main_yaml)

        print(f"Generated configs for {dataset_id}  -> {split_name}")

    if not found_any:
        print(f"Skipping {subdir}: no supported split folder found "f"({', '.join(SPLIT_DIRS.keys())}).")
