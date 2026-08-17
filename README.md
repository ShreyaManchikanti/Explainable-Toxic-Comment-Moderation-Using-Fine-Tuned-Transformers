# Explainable Toxic Comment Moderation

Masters coursework project: a multi-label toxic comment classifier (BERT and RoBERTa,
fine-tuned and compared against a TF-IDF + Logistic Regression baseline), with SHAP
explainability and a simple keyword-based bias audit, plus a Gradio demo.

Dataset: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
(already included as `data/jigsaw-toxic.zip`).
## Download Data and Models

- **Data:** [Download the Data Folder](https://drive.google.com/drive/folders/1Bg5jquxWE6XwjluGgfN-G9MDxK0FU_fK?usp=sharing)

- **Models:** [Download the Models Folder](https://drive.google.com/drive/folders/1NUR_JFodAMswSsz1yn6EHcve6XloUuVg?usp=sharing)

> **Important:** After downloading, place both the `data` and `models` folders directly inside the main project folder:
>
> `Advance-Artificial-Intelligence-Project(Group3)/`
>
> The final structure should be:
>
> ```text
> Advance-Artificial-Intelligence-Project(Group3)/
> ├── data/
> ├── models/
> ├── notebooks/
> ├── reports/
> ├── app.py
> ├── requirements.txt
> └── README.md
> ```
## Project structure

```
data/
  jigsaw-toxic.zip          - original dataset download
  raw/                      - extracted CSVs (created by setup, gitignored)
  processed/                - subsamples + saved model predictions (created by the notebooks)
notebooks/
  01_data_exploration.ipynb          - label distribution, class imbalance, comment length
  02_baseline_tfidf_logreg.ipynb     - builds the shared working subsample + trains the baseline
  03_finetune_bert_roberta.ipynb     - fine-tunes BERT and RoBERTa on the subsample
  04_evaluate_and_compare.ipynb      - scores all three models on the held-out test set
  05_shap_explainability.ipynb       - SHAP token-attribution explanations
  06_bias_audit.ipynb                - keyword-based identity-term false-positive-rate audit
models/                       - saved fine-tuned checkpoints (gitignored, regenerate via notebook 03)
reports/
  metrics.json                - all models' metrics, built up notebook by notebook
  *.png                       - saved comparison/audit charts
app.py                        - Gradio demo
requirements.txt
```

## Setup

Requires Python 3.11 (PyTorch doesn't yet support 3.14, which is why the venv is pinned).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then extract the dataset once:

```bash
mkdir -p data/raw
unzip -j data/jigsaw-toxic.zip "jigsaw-toxic/*.csv" -d data/raw
```

## Running the project

Run the notebooks **in order** - each one depends on files written by the previous ones:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_tfidf_logreg.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_finetune_bert_roberta.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluate_and_compare.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/05_shap_explainability.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/06_bias_audit.ipynb
```

(Or just open them in Jupyter and run all cells - `.venv/bin/jupyter notebook`.)

Notebook 03 (fine-tuning both transformers) is the slow one - expect it to take a while
even on a laptop GPU (Apple Silicon MPS). Everything else runs in a couple of minutes.

Then launch the demo:

```bash
.venv/bin/python app.py
```

**Quickstart for teammates who already have `models/` (or a `models.zip`/`MODEL_ARCHIVE_URL`)
and just want the UI, no notebooks:**

```bash
python3.11 -m venv .venv          # first time only
.venv/bin/python run_app.py
```

`run_app.py` installs anything missing from `requirements.txt`, checks that the
winning model's weights (`models/<bert|roberta>/`, gitignored - see below) and
`reports/metrics.json` are present, and then launches `app.py`. It doesn't train
anything; if the model weights aren't there it looks for `models.zip` in the repo
root or a `MODEL_ARCHIVE_URL` env var to fetch them from, and fails with clear
instructions if neither is available.

## Key design choices (and why)

This project deliberately keeps things simple, given the scope:

- **Working subsample, not the full 159k-row training set**: ~10,000 comments (half
  toxic, half clean) are used to fine-tune BERT/RoBERTa so training finishes in minutes,
  not hours, on a laptop. The baseline is trained on the exact same subsample so the
  comparison is fair. All models are then *evaluated* on the full, untouched official
  test set (~64k rows).
- **Class imbalance handled by subsampling, not SMOTE**: the 50/50 toxic/clean subsample
  already addresses most of the imbalance problem; we didn't add embedding-level SMOTE
  on top, since it has no clean semantics for multi-label text.
- **Bias audit uses a keyword-proxy, not real identity annotations**: this Jigsaw dataset
  doesn't include identity-group labels (those exist in a separate, larger Jigsaw
  dataset). Instead, notebook 06 checks whether comments *mentioning* certain identity
  words get flagged toxic at a higher rate than average - a simplified stand-in for a
  full fairness audit.
- **No `fairlearn` / Equalised-Odds metrics**: false-positive-rate comparisons via plain
  pandas cover the same idea with far less new API surface to learn and explain.

These are noted as intentional scope decisions in the accompanying report, not
oversights.
