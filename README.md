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

### 1. Download raw corpus (or add your own)

```bash
python3 -m data_processing.download_babylm
```

Or manually put your `.txt` file in:

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

### 5. Inject animacy-based case markers

```bash
python3 -m perturbation.perturb --mode heuristic
python3 -m perturbation.perturb --mode rule
```

This generates three outputs in the folder `data/perturbed/{mode}/`:

- `*_affected.txt`: sentences with case marking added
- `*_unaffected.txt`: structurally valid but no perturbation needed
- `*_invalid.txt`: no usable verb structure found

Replace `{mode}` with either `rule` or `heuristic` depending on the chosen strategy.

---

### 6. Generate human check examples
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