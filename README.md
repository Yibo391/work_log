# SemEval 2026 Task 9 — Subtask 1 (Polarization Detection) — Work Log

This repo tracks my experiments for **SemEval 2026 Task 9 / Subtask 1**: binary classification of whether a text contains **attitude polarization** (0/1) across **22 languages**.

## Dataset layout (as used in Colab)
I downloaded the official dataset and moved it to Colab:

- `/content/train/` — training CSVs, one per language (`amh.csv`, `deu.csv`, ..., `zho.csv`)
- `/content/dev/` — dev CSVs, one per language (often **unlabeled** in the challenge setup)

Each language file contains at least:
- `id` (string)
- `text` (string)
- `polarization` (label, may be missing depending on split)

> Important: In this shared task, the public **dev** may have empty labels.  
> Therefore, I evaluate using **held-out splits from train**.

---

## What I have implemented

### A) Deep learning baseline (multilingual fine-tuning)
Notebook: `dev_pharse.ipynb`

- Model: `xlm-roberta-base` fine-tuned for binary classification.
- Data handling:
  - Loads all train language CSVs
  - Robust label normalization (casts strings/NaNs safely)
  - Drops rows with missing labels for training (keeps only 0/1)
- Evaluation:
  - Since dev labels can be empty, creates a fixed **train/val split** from training data.
  - Computes **Macro F1** (official metric) on that held-out validation split.
  - Optional: threshold tuning to slightly improve Macro F1.

Result observed:
- Macro F1 around **0.91** on the held-out split (NOT the official hidden test).

---

### B) Translation → Empath → Logistic Regression (all 22 languages)
Notebook: `empath.ipynb`

Goal:
- Translate each language to English using NLLB
- Extract Empath category features (194 dims)
- Train Logistic Regression and evaluate Macro F1

Status:
- Pipeline structure is correct, but **translation is extremely slow** for 22 languages.
- Also hit runtime issues (translation tokenizer/lang-code mismatch and pandas dtype assignment errors).
- Needs optimization (batching, caching, and possibly running fewer languages or smaller sample).

---

### C) German-only Translation → Empath → Logistic Regression
Notebook: `Untitled4.ipynb`

- Scope limited to German (`deu.csv`) to make CPU translation feasible.
- Translation: Argos Translate (CPU)
- Features: Empath
- Classifier: Logistic Regression
- Result: Macro F1 ~ **0.50** (close to weak baseline)

Explanation (high-level):
- Empath features are coarse and not well-aligned with polarization cues.
- Translation may wash out slang/targeting/group markers that matter for polarization detection.
- 194-dimensional Empath vectors alone often underfit this task.

---

## Why results differ a lot
- The deep learning approach learns directly from multilingual text and captures semantics; its val score is on a **held-out train split**, which can be optimistic.
- Empath-only baselines are much weaker because they compress text into broad lexical categories and lose key polarization signals.

---

## Next steps (planned)
1. Add stronger classical baselines:
   - Char n-gram TF-IDF + Linear SVM / Logistic Regression
   - Combine TF-IDF + Empath (feature concatenation)
2. Data enlargement:
   - Translation-based augmentation (back-translation / paraphrase)
---

### D) Paper review (examplepaper.pdf) — applicability to POLAR Subtask 1
- The paper studies time-aware prediction from sequences of posts (forecasting next label from prior posts).
- This does **not directly apply** to POLAR Subtask 1 because our dataset is single-text binary classification with no user-timeline structure.
- What does transfer is the **two-stage design idea**:
  - strong per-text classifier (e.g., XLM-R)
  - lightweight post-processing (threshold calibration / stacking with classical features).
- The paper does not propose dataset synthesis; augmentation for this task must be designed separately (e.g., back-translation, counterfactual edits).

## Notes on evaluation (Macro F1)
The shared task uses **Macro F1** for binary labels (0/1).
If dev labels are empty, we estimate performance using a held-out split from train.

---

## Files
- `dev_pharse.ipynb` — multilingual deep learning fine-tuning (XLM-R)
- `empath.ipynb` — translation→Empath (22 languages; slow and currently buggy)
- `Untitled4.ipynb` — German-only translation→Empath baseline (poor performance)
