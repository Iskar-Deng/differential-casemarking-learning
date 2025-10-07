# Semantic Triggers and Grammatical Learnability: Probing Differential Argument Marking in LMs

> **Note:** This repository is an *in-progress* research implementation.  
> Some components (e.g., classifier training and perturbation injection) are experimental and under active development.  
> The current working branch is **`no-mistral`**, which contains a complete pipeline from corpus filtering to model training,  
> though stability and full reproducibility are *not yet guaranteed*.

![Skynet Skyboy](./skynet-skyboy.gif)


## Task Summary

We construct an English corpus that simulates a Naxi-style relational case system, where overt case marking depends on the animacy comparison between subject and object. This experiment tests whether small language models can learn such typologically rare but cognitively plausible patterns under limited supervision.

---

## Datasets

We ues BabyLM, available at: [https://babylm.github.io/](#)

---

## Setup

### 1. Install Dependencies
```bash
conda env create -f environment.yml
```

Disabled:
```bash
pip install -r requirements.txt
pip install spacy benepar torch transformers tqdm protobuf==3.20.*
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('wordnet')"
python -c "import benepar; benepar.download('benepar_en3')"
```

---

### 2. Configure Paths

Edit `utils.py` and set the following variables:  

```python
# Path to datasets
DATA_PATH = "/absolute/path/to/data"

# Path to store trained models
MODEL_PATH = "/absolute/path/to/models"

# Path to cloned Mistral framework folder
MISTRAL_PATH = "/absolute/path/to/mistral"

# Paths for cache and checkpoints (can be any writable local folder)
CACHE_PATH = "/absolute/path/to/cache"
CHECKPOINT_PATH = "/absolute/path/to/checkpoints"

# Config and evaluation paths (use absolute paths from cloned repo)
CONFIG_PATH = "/absolute/path/to/mistral/configs"
EVALUATION_PATH = "/absolute/path/to/mistral/evaluation"
```

**Notes:**
- `DATA_PATH` is where your datasets (e.g., BabyLM) are stored.
- `MODEL_PATH`, `CACHE_PATH`, `CHECKPOINT_PATH` can be any local directories for saving outputs and temporary files.
- `CONFIG_PATH` and `EVALUATION_PATH` should point to the **configs** and **evaluation** directories in this repo.
- `MISTRAL_PATH` is the root folder of your cloned `mistral` repository.

## Run the Pipeline

### 1. Prepare raw corpus 

Put your `.txt` training file in:

```
<DATA_PATH>/raw/your_data.txt
```

Make sure only one `.txt` file exists under `raw/`.

---

### 2. Filter noisy or irrelevant sentences
```bash
python -m data_processing.split_corpus
python -m data_processing.filter_sentences
```

---

### 3. Parse using spaCy + benepar

```bash
python -m perturbation.parse
```

---

### 4. Extract verb arguments and spans

```bash
python3 -m perturbation.extract_verb
```

---

### 5. Train Animacy Classifer (Set your openai api into env first)
```bash
python -m classifiers.generate_data --task animacy --max 5000 --strategy random
python -m classifiers.generate_data --task npdef --max 5000 --strategy balanced           
python -m classifiers.train_classifier --task animacy --epochs 10 --amp
python -m classifiers.train_classifier --task npdef --epochs 10 --amp
```

---

### 6. Inject animacy-based case markers

```bash
python -m perturbation.run_perturb \
  --strategy local \
  --A_mode none --A_markedness forward \
  --P_mode animal-p12 --P_markedness forward-inverse
```

---

### 8. Prepare the config for training
```bash
python -m tools.generate_yaml --mode rule --strategy A+P --overwrite 
``` 

---

### 9. Train the model
```bash
python -m training.train_lm --mode rule --strategy A+P  
``` 

### 10. Experiment 1 - ppl
```bash
python -m evaluation.eval_ppl --run-id rule_A+P
python -m evaluation.eval_ppl --run-id rule_full
...
MPLBACKEND=Agg python evaluation/plot_ppl_curves.py \
    --runs rule_A+P rule_A+P_with_invalid rule_none rule_none_with_invalid rule_full \
    --out results/ppl_comparison.png
```             
---

### 11. Experiment 2 - Minipairs

#### Prepare the pairs
```bash
python -m evaluation.perturb_blimp_pairs \
  --blimp evaluation/BLiMP_raw/animate_subject_trans.jsonl \
  --out evaluation/BLiMP_perturbed/animate_subject_trans \
  --mode rule \
  --strategy A+P

python -m evaluation.casemarking.generate_minipair --mode rule --strategy A+P --limit 1000 
```

#### Eval the pairs
```bash
python -m evaluation.eval_minipairs \
    --run-id rule_A+P \
    --jsonl evaluation/casemarking/rule_A+P/cbt_minimal_pairs.jsonl \
    --out-dir results_raw_mp
```

#### Single Eval
```bash
python -m evaluation.eval_single_ckpt \
  --checkpoint /home/hd49/relational-casemarking-learning/checkpoints/local_Anone-forward_Pp3-forward/checkpoint-50000 \
  --jsonl /home/hd49/relational-casemarking-learning/evaluation/BLiMP_raw/regular_plural_subject_verb_agreement_1.jsonl \
  --batch-size 16 \
  --fp16 \
  --save-details results/1.jsonl

python -m evaluation.eval_single_ckpt \
  --checkpoint checkpoints/local_Anone-forward_Pp3-forward/checkpoint-3000 \
  --jsonl /home/hd49/relational-casemarking-learning/evaluation/BLiMP_raw/regular_plural_subject_verb_agreement_1.jsonl \
  --batch-size 16 \
  --fp16 \
  --save-details results/Pp3.jsonl

```

#### Draw the plot
```bash
MPLBACKEND=Agg python evaluation/plot_minipair_accuracy.py \
    --result-dir results_blipmp/animate_subject_trans \
    --out plots/animate_subject_trans_accuracy.png
```