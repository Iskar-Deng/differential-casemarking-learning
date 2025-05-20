# Learning What Models Can't: Animacy-Based Case Marking in Natural Language

This repository contains the data, scripts, and evaluation materials for our Ling 575 project on testing neural models’ ability to learn relational animacy-based case systems using English rewrites.

## 📂 Structure

- `data/`: Raw and processed datasets
- `scripts/`: Code for parsing, rewriting, splitting, and evaluating
- `templates/`: Rule documents and edge case logs
- `evaluation/`: Analysis and visualizations
- `reports/`: Project documents and final write-up

## 📋 Task Summary

We construct an English corpus rewritten to simulate a Naxi-style relational case system, where overt case marking depends on comparing the animacy of subject and object. We evaluate whether small language models can learn this rule under limited input conditions.

## ⚙️ Setup

```bash
pip install -r requirements.txt
```
