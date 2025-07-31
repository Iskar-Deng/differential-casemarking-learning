# Learning What Models Can't: Animacy-Based Case Marking in Natural Language

![Skynet Skyboy](./skynet-skyboy.gif)


## Task Summary

We construct an English corpus that simulates a Naxi-style relational case system, where overt case marking depends on the animacy comparison between subject and object. This experiment tests whether small language models can learn such typologically rare but cognitively plausible patterns under limited supervision.

---

## Datasets

From the BabyLM 100M clean subset:

> https://huggingface.co/datasets/cambridge-climb/BabyLM/tree/main/clean/100M  
We use: `cbt.txt`

---

## Setup

```bash
pip install -r requirements.txt
pip install spacy benepar torch transformers tqdm protobuf==3.20.*
python -m spacy download en_core_web_trf
python -c "import nltk; nltk.download('wordnet')"
python -c "import benepar; benepar.download('benepar_en3')"
```

Set your data path by editing `utils.py`:

```python
DATA_PATH = "/your/local/data/directory"
```

---

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
python3 -m data_processing.filter_sentences
```

---

### 3. Parse using spaCy + benepar

```bash
python3 -m perturbation.parse
```

---

### 4. Extract verb arguments and spans

```bash
python3 -m perturbation.extract_verb
```

---

### 5. Train Animacy Classifer
```bash
python3 -m animacy_classifer.extract_verb
python -m animacy_classifer.train_classifer   
```

---

### 6. Inject animacy-based case markers

```bash
python -m perturbation.perturb_with_model --mode rule --strategy A+P 
python -m perturbation.perturb_with_model --mode heuristic --strategy A+P
python -m perturbation.perturb_with_model --mode rule --strategy A_only
python -m perturbation.perturb_with_model --mode rule --strategy P_only
python -m perturbation.perturb_with_model --mode rule --strategy none
python -m perturbation.perturb_with_model --mode rule --strategy full
```

---

### 7. Generate train and vad
```bash
python -m data_processing.generate_vad
``` 

---

### 8. Prepare BLiMP
```bash
python -m evaluation.perturb_blimp_pairs \
  --blimp evaluation/BLiMP_raw/regular_plural_subject_verb_agreement_1.jsonl \
  --out evaluation/BLiMP/regular_plural_subject_verb_agreement_1.jsonl \
  --mode rule \
  --strategy A+P
```

### Select human check examples
```bash
python3 -m data_processing.human_spot_check --num_lines 50 --seed 42
```

## Structure

```
.
├── data_processing/
│   ├── download_babylm.py         # Download BabyLM corpus
│   ├── filter_sentences.py        # Filter sentences by length
│   └── human_spot_check.py        # Randomly select lines for spot-check
├── perturbation/
│   ├── parse.py                   # Run spaCy + benepar parsing
│   ├── extract_verb.py            # Extract SVO and span structures
│   └── perturb.py                 # Inject case markers based on animacy
├── utils.py                       # Global path / markers / animacy functions
├── requirements.txt
└── ...
```