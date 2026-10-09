# GitLab Commit Guide — Riya & Praveen

The course requires **commit history from both members**, each in their lead role. This guide takes you from an empty GitLab project to a pushed repository in which GitLab shows two contributors.

> **Rule:** each person commits **their own files, from their own GitLab account, on their own machine/session**. Do not share tokens or commit under someone else's name — the history is evidence of who did what, and both of you are examined on it in the viva. Read and re-run your part before committing it.

---

## 0. Who commits what

| Riya — Model Lead | Praveen — Application Lead | Shared |
|---|---|---|
| `src/config.py`, `data_preprocessing.py`, `numpy_perceptron.py`, `model.py`, `focal_loss.py`, `training.py`, `metrics.py`, `calibration.py`, `cost_optimizer.py`, `plotting.py` | `app/` (main.py, logger.py, templates, static) | `notebooks/07_final_evaluation.ipynb` |
| `notebooks/01` – `06` | `src/inference.py` | `src/explainability.py` |
| `data/`, `models/`, `plots/`, `results/` | `Dockerfile`, `.dockerignore`, `tests/test_api.py`, `logs/` | `report/`, `README.md`, `docs/` |
| `tests/` for the model code | `docs/DEPLOY_HUGGINGFACE.md`, `docs/hf_space_README.md` | `requirements.txt`, `.gitignore` |

---

## 1. One-time setup (each person, on their own computer)

**1.1 Tell git who you are** — use the same e-mail as your GitLab account, otherwise GitLab cannot link the commits to you:
```bash
git config --global user.name "Your Full Name"
```
```bash
git config --global user.email "your-gitlab-email@example.com"
```

**1.2 Authenticate to GitLab** — choose one:

* **SSH key (recommended).** Create a key, then paste the public key into GitLab → *Avatar → Edit profile → SSH Keys*:
```bash
ssh-keygen -t ed25519 -C "your-gitlab-email@example.com"
```
```bash
cat ~/.ssh/id_ed25519.pub
```
```bash
ssh -T git@gitlab.com
```
  The last command should answer `Welcome to GitLab, @yourname!`.

* **Personal access token (HTTPS).** GitLab → *Avatar → Edit profile → Access tokens* → create a token with the `write_repository` scope. Git asks for it as the password the first time you push. Keep it private: never put it in a file, a notebook or a chat.

---

## 2. Create the GitLab project (one person, once)

1. GitLab → **New project → Create blank project**.
2. Name: `telco-churn-dashboard` · Visibility: as your instructor requires · **untick "Initialize repository with a README"** (we already have one).
3. **Manage → Members → Invite members** → add your teammate with the **Maintainer** role (Developers cannot push to the protected `main` branch by default).

---

## 3. Riya: create the repository and push the model work

From the project folder (`telco-churn-dashboard/`):

```bash
git init -b main
```
```bash
git remote add origin git@gitlab.com:<your-group-or-username>/telco-churn-dashboard.git
```
(HTTPS alternative: `https://gitlab.com/<your-group-or-username>/telco-churn-dashboard.git`)

Commit in small, meaningful steps — one topic per commit:

```bash
git add .gitignore requirements.txt src/__init__.py src/config.py && git commit -m "Project skeleton: config, seeds, dependencies"
```
```bash
git add data/ src/data_preprocessing.py src/plotting.py notebooks/01_eda.ipynb notebooks/02_preprocessing.ipynb tests/conftest.py tests/test_preprocessing.py && git commit -m "EDA and leakage-free preprocessing (fit on train only, 60/20/20 split)"
```
```bash
git add src/numpy_perceptron.py src/metrics.py notebooks/03_baselines.ipynb tests/test_numpy_perceptron.py && git commit -m "Baselines: logistic regression and from-scratch NumPy perceptron with gradient check"
```
```bash
git add src/model.py src/training.py notebooks/04_mlp_training.ipynb tests/test_training.py && git commit -m "PyTorch MLP with BatchNorm, Dropout, L2, early stopping; optimiser comparison"
```
```bash
git add src/focal_loss.py notebooks/05_imbalance_experiments.ipynb tests/test_focal_loss.py && git commit -m "Imbalance ablation: plain, weighted BCE, oversampling, from-scratch focal loss"
```
```bash
git add src/calibration.py src/cost_optimizer.py notebooks/06_cost_threshold.ipynb tests/test_cost_optimizer.py && git commit -m "Platt calibration and cost-based threshold optimisation"
```
```bash
git add models/ plots/ results/ && git commit -m "Trained artefacts, figures and result tables"
```
```bash
git push -u origin main
```

---

## 4. Praveen: clone, add the application work, push

Praveen needs the application files that are not yet in the repository (copy them from the shared project folder into his clone — or work in the same Drive folder with **his own** git identity and credentials).

```bash
git clone git@gitlab.com:<your-group-or-username>/telco-churn-dashboard.git && cd telco-churn-dashboard
```
```bash
git add src/inference.py app/__init__.py app/logger.py && git commit -m "Inference service (ChurnPredictor) and prediction logger (JSONL + SQLite)"
```
```bash
git add app/main.py tests/test_api.py && git commit -m "FastAPI backend: predict, explain, logs, clear-log endpoints with validation"
```
```bash
git add app/templates app/static && git commit -m "Dashboard UI: upload, cost planner, charts, risk table, customer drawer, icons"
```
```bash
git add Dockerfile .dockerignore run_app.sh run_app.bat START_HERE.md docs/DEPLOY_HUGGINGFACE.md docs/hf_space_README.md logs/ && git commit -m "Deployment: Dockerfile and Hugging Face Spaces guide"
```
```bash
git push origin main
```

---

## 5. Shared files (either member; ideally split them)

Always pull first so you do not overwrite each other:
```bash
git pull --rebase origin main
```
```bash
git add src/explainability.py notebooks/07_final_evaluation.ipynb && git commit -m "Final single test-set evaluation and SHAP explainability"
```
```bash
git add README.md docs/GITLAB_COMMIT_GUIDE.md && git commit -m "README and commit guide"
```
```bash
git add report/ && git commit -m "Report and viva guide"
```
```bash
git push origin main
```
Suggestion: one of you commits notebook 07 + explainability, the other the report, so both have commits on shared work too.

---

## 6. Verify both contributors appear

* GitLab → **Code → Commits**: every commit shows the right author.
* GitLab → **Analyze → Contributor analytics** (or *Repository graph*): two contributors.
* Locally:
```bash
git shortlog -sne
```
```bash
git log --pretty=format:"%h  %an  %s" --name-status | head -60
```
If a commit shows the wrong name, the `user.email` on that machine does not match the GitLab account — fix it (step 1.1) **before** making more commits.

---

## 7. Before the final push — checklist

- [ ] `python -m pytest -q` passes (49 tests).
- [ ] `.venv/`, `__pycache__/` and `.DS_Store` are **not** staged (`git status` — they are in `.gitignore`).
- [ ] `models/*.pth`, `models/*.pkl`, `models/model_config.json`, `models/shap_background.npy` **are** committed (a fresh clone must run the web app).
- [ ] No token, password or `.env` file anywhere: `git grep -n -i "token\|password\|secret"` shows nothing sensitive.
- [ ] The prediction log is in the state you want (clear it with the dashboard's **Clear log** button before the demo).
- [ ] `report/report.pdf` exported and committed.
- [ ] Clone into a new folder and run `uvicorn app.main:app` to confirm it works from scratch.

---

## 8. Everyday commands

| Task | Command |
|---|---|
| See what changed | `git status` · `git diff` |
| Get your teammate's work | `git pull --rebase origin main` |
| Undo an un-committed change to one file | `git restore <file>` |
| Unstage a file | `git restore --staged <file>` |
| Fix the last commit message (before pushing) | `git commit --amend -m "new message"` |
| Push rejected ("fetch first") | `git pull --rebase origin main` then `git push` |
| Merge conflict in a notebook | keep one version (`git checkout --theirs <nb>` or `--ours`), re-run the notebook, commit |

**Commit message style:** short imperative summary of *what* and *why* — e.g. "Add Platt calibration so risk tiers reflect real probabilities", not "update files".
