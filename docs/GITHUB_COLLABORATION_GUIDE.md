# GitHub Collaboration Guide — Praveen & Riya

**Project:** P5. Telecom Churn Early-Warning Dashboard  
**Repository:** [https://github.com/fridayslifes/NNDL](https://github.com/fridayslifes/NNDL)  
**Requirement:** Commit history from **both members** in their assigned lead roles.

---

## 0. Role Division & Commit Ownership

| Member | Role | Files Owned & Committed |
|---|---|---|
| **Praveen** (`fridayslifes`) | **Application Lead** | `app/` (FastAPI backend `main.py`, `logger.py`, `templates/`, `static/`), `src/inference.py`, `tests/test_api.py`, `Dockerfile`, `.dockerignore`, `run_app.sh`, `run_app.bat`, `START_HERE.md`, `docs/DEPLOY_HUGGINGFACE.md`, `docs/hf_space_README.md`, `logs/.gitkeep`, initial project skeleton (`.gitignore`, `requirements.txt`, `README.md`). |
| **Riya** | **Model Lead** | `data/` (`Telco-Customer-Churn.csv`, `sample_demo.csv`, `processed/`), `notebooks/01_eda.ipynb` to `06_cost_threshold.ipynb`, `src/config.py`, `src/data_preprocessing.py`, `src/plotting.py`, `src/numpy_perceptron.py`, `src/metrics.py`, `src/model.py`, `src/training.py`, `src/focal_loss.py`, `src/calibration.py`, `src/cost_optimizer.py`, `models/`, `plots/`, `results/`, `tests/` (model tests). |
| **Shared / Joint** | **Both** | `notebooks/07_final_evaluation.ipynb`, `src/explainability.py`, `report/` (`report.md`, `viva_guide.md`). |

---

## 1. Praveen's Setup (Completed)

Praveen has pushed the initial application skeleton, API backend, dashboard UI, and deployment configs to `https://github.com/fridayslifes/NNDL.git`.

### Next Step for Praveen on GitHub:
1. Go to repository **Settings** → **Collaborators** (`https://github.com/fridayslifes/NNDL/settings/access`).
2. Click **Add people**.
3. Search for Riya's GitHub username or email and send the invite.
4. Riya accepts the invitation via email or at `https://github.com/fridayslifes/NNDL/invitations`.

---

## 2. Step-by-Step Instructions for Riya (Model Lead)

Riya runs these steps on **her own machine/session** using **her own GitHub account** so her commits are registered under her identity.

### Step 2.1: Configure Git Identity
Riya must ensure her git name and email match her GitHub account:
```bash
git config --global user.name "Riya"
git config --global user.email "riya-github-email@example.com"
```

### Step 2.2: Clone the Repository
```bash
git clone https://github.com/fridayslifes/NNDL.git
cd NNDL
```

### Step 2.3: Add Model Files into the Repository
Copy the model pipeline files, notebooks, data, and trained weights from the shared project folder into this cloned `NNDL/` directory:
- `data/`
- `notebooks/` (01 to 06)
- `src/` (`config.py`, `data_preprocessing.py`, `numpy_perceptron.py`, `model.py`, `training.py`, `focal_loss.py`, `calibration.py`, `cost_optimizer.py`, `plotting.py`, `metrics.py`)
- `tests/` (`conftest.py`, `test_preprocessing.py`, `test_numpy_perceptron.py`, `test_training.py`, `test_focal_loss.py`, `test_cost_optimizer.py`)
- `models/`
- `plots/`
- `results/`

### Step 2.4: Stage and Commit in Logical Milestones
Riya should commit her work in clean, logical commits representing each stage of the ML pipeline:

#### Commit 1: Data Pipeline & Preprocessing
```bash
git add data/ src/config.py src/data_preprocessing.py src/plotting.py notebooks/01_eda.ipynb notebooks/02_preprocessing.ipynb tests/conftest.py tests/test_preprocessing.py
git commit -m "Data pipeline: cleaning, leakage-free preprocessing (60/20/20 train/val/test), and EDA"
```

#### Commit 2: Baselines (Logistic Regression & Perceptron)
```bash
git add src/numpy_perceptron.py src/metrics.py notebooks/03_baselines.ipynb tests/test_numpy_perceptron.py
git commit -m "Baselines: logistic regression and from-scratch NumPy perceptron with gradient checking"
```

#### Commit 3: Deep Learning MLP Architecture
```bash
git add src/model.py src/training.py notebooks/04_mlp_training.ipynb tests/test_training.py
git commit -m "PyTorch MLP classifier with BatchNorm, Dropout, L2 decay, early stopping and optimizer benchmarking"
```

#### Commit 4: Class Imbalance Ablation
```bash
git add src/focal_loss.py notebooks/05_imbalance_experiments.ipynb tests/test_focal_loss.py
git commit -m "Class imbalance ablation: plain BCE vs weighted BCE vs oversampling vs from-scratch focal loss"
```

#### Commit 5: Calibration & Cost-Based Threshold
```bash
git add src/calibration.py src/cost_optimizer.py notebooks/06_cost_threshold.ipynb tests/test_cost_optimizer.py
git commit -m "Probability calibration (Platt scaling) and business cost-based threshold optimization"
```

#### Commit 6: Trained Weights, Plots, and Result Tables
```bash
git add models/ plots/ results/
git commit -m "Artefacts: trained weights, calibration parameters, evaluation plots, and ablation metrics"
```

### Step 2.5: Push to GitHub
```bash
git pull --rebase origin main
git push origin main
```
*(If prompted for credentials, use your GitHub username and personal access token or SSH key.)*

---

## 3. Joint / Shared Work (Final Evaluation & Report)

Either Praveen or Riya (or both together) can commit the remaining joint items:
```bash
# Pull latest changes first
git pull --rebase origin main

# Add final evaluation and explainability
git add src/explainability.py notebooks/07_final_evaluation.ipynb
git commit -m "Final evaluation: single test set run, bootstrap confidence intervals, and SHAP explainability"

# Add project report and viva preparation guide
git add report/
git commit -m "Project documentation: comprehensive technical report and viva preparation guide"

git push origin main
```

---

## 4. How to Verify Both Contributors on GitHub

1. Open [https://github.com/fridayslifes/NNDL](https://github.com/fridayslifes/NNDL).
2. Look at the right sidebar under **Contributors**: both **fridayslifes** and **Riya** should appear.
3. Click on the commit history (`Commits` tab): you should see distinct commits authored by Praveen and distinct commits authored by Riya.
4. Locally, run:
```bash
git shortlog -sne
```
Output will confirm:
```
     X  Praveen <...>
     Y  Riya <...>
```

---

## 5. Daily Git Workflow & Best Practices

- **Always pull before starting work:**
  ```bash
  git pull --rebase origin main
  ```
- **Check changed files before adding:**
  ```bash
  git status
  ```
- **Never commit secrets or tokens:**
  Never put API keys or tokens in code or notebooks.
- **Merge conflicts in notebooks:**
  If a notebook has a merge conflict, keep one copy (`git checkout --ours notebook.ipynb` or `--theirs`), open it in Jupyter/VS Code, execute cells from top to bottom, save, and commit.
