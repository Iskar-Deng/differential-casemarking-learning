# utils.py

# === Constants ===

DATA_PATH = "/home/hd49/relational-casemarking-learning/data"
MODEL_PATH = "/home/hd49/relational-casemarking-learning/models"
CONFIG_PATH = "/home/hd49/relational-casemarking-learning/configs"
MISTRAL_PATH = "mistral"
CHECKPOINT_PATH = "/home/hd49/relational-casemarking-learning/checkpoints"
CACHE_PATH = "/home/hd49/relational-casemarking-learning/cache"
EVALUATION_PATH = "evaluation"

AGENT_MARK = "🄰"
PATIENT_MARK = "🄿"

ANIMACY_RANK = {
    "human": 1,
    "animal": 2,
    "inanimate": 3
}

PRONOUNS_HUMAN = {
    "i", "me", "you", "he", "him", "she", "her", "we", "us", "they", "them",
    "myself", "yourself", "himself", "herself", "ourselves", "yourselves", "themselves"
}

# === Animacy Utilities ===
# Hierarchy: Human > Animal > Inanimate

import nltk
from nltk.corpus import wordnet as wn

def clean_phrase(phrase: str) -> str:
    tokens = phrase.lower().strip().split()
    if tokens and tokens[0] in {"a", "an", "the", "this", "that", "my", "your", "his", "her", "their"}:
        tokens = tokens[1:]
    return tokens[-1] if tokens else phrase


def get_animacy_category(phrase: str) -> str:
    word = clean_phrase(phrase)

    if word in PRONOUNS_HUMAN:
        return "human"

    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return "inanimate"

    syn = synsets[0]
    hypernyms = {lemma.name() for s in syn.closure(lambda s: s.hypernyms()) for lemma in s.lemmas()}
    print(f"[DEBUG] Word: {word} | Hypernyms: {hypernyms}")
    if "person" in hypernyms or "people" in hypernyms:
        return "human"
    elif "animal" in hypernyms or "organism" in hypernyms:
        return "animal"
    else:
        return "inanimate"


def compare_animacy(phrase1: str, phrase2: str) -> str:
    a1 = get_animacy_category(phrase1)
    a2 = get_animacy_category(phrase2)
    r1 = ANIMACY_RANK[a1]
    r2 = ANIMACY_RANK[a2]

    if r1 < r2:
        return "higher"
    elif r1 > r2:
        return "lower"
    else:
        return "equal"
    
def is_human_subject(phrase: str) -> bool:
    """Return True if the subject is human based on category."""
    return get_animacy_category(phrase) == "human"

def should_perturb_heuristic(subject_phrase: str) -> bool:
    """Heuristic: Add case marker if subject is NOT human."""
    return not is_human_subject(subject_phrase)