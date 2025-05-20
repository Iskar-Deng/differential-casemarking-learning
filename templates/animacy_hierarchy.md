# Revised DOM Rules with Postpositional Case Markers

## 📘 Rule Set 1: Simplified Animacy-Only DOM (3 Levels)

In this rule set, DOM is based on a simplified animacy scale with **three levels**:

### 🔢 Animacy Levels (from highest to lowest):

```
1. Human
2. Animal
3. Inanimate
```

### 🧾 DOM Rule:
- If the **subject's animacy is not higher** than the object's animacy, then:
  - Add `AGE` to the subject
  - Add `PAT` to the object
- Otherwise, no marking is applied.

### ✅ Examples:

| Sentence | Subject Animacy | Object Animacy | Rewritten |
|----------|------------------|----------------|-----------|
| I met **a teacher**. | Human | Human | I AGE met a teacher PAT. |
| The dog bit **the man**. | Animal | Human | The dog AGE bit the man PAT. |
| The teacher praised **the dog**. | Human | Animal | The teacher praised the dog. |
| The robot pushed **a car**. | Inanimate | Inanimate | The robot AGE pushed a car PAT. |
| A lion chased **a zebra**. | Animal | Animal | A lion AGE chased a zebra PAT. |

---

## 📗 Rule Set 2: Combined Animacy + Definiteness DOM

This rule combines **animacy** and **definiteness** for a more nuanced marking system.

### 🔢 Combined Hierarchy:

```
1st/2nd Person Pronoun > 3rd Person Pronoun > Human Proper Noun > Definite Human NP > Specific Human Indefinite > Definite Animal NP > Inanimate Definite > Inanimate Indefinite
```

### 🧾 DOM Rule:
- Determine the **prominence** of subject and object based on combined scale.
- If the subject is **not more prominent** than the object:
  - Add `AGE` to the subject
  - Add `PAT` to the object
- Otherwise, no markers are used.

### ✅ Examples:

| Sentence | Subject Type | Object Type | Rewritten |
|----------|--------------|-------------|-----------|
| I met **Mary**. | 1st person pronoun | Proper noun | I met Mary. |
| Mary met **me**. | Proper noun | 1st person pronoun | Mary AGE met me PAT. |
| The teacher saw **a lion**. | Definite Human NP | Definite Animal NP | The teacher saw a lion. |
| A dog chased **the teacher**. | Animal | Definite Human NP | A dog AGE chased the teacher PAT. |
| A machine moved **a box**. | Inanimate Indefinite | Inanimate Indefinite | A machine AGE moved a box PAT. |

---

## ✍️ Notes:
- `AGE` marks the agent if it is not more prominent (or higher in animacy) than the object.
- `PAT` marks the object if the agent is not more prominent than it.
- This logic models DOM based on **relational prominence**, not absolute position in the scale.
- These rules can be used in typological modeling, constructed languages, or experimental syntax systems.