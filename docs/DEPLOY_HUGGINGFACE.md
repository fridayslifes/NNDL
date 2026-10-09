# Deploying to Hugging Face Spaces with Docker

**Owner:** Praveen (Application Lead)

This publishes the dashboard at `https://huggingface.co/spaces/<username>/<space-name>` using the project's `Dockerfile`. The free **CPU basic** hardware (2 vCPU, 16 GB RAM) is more than enough: the model has 4,929 parameters and inference takes milliseconds.

> **Status of this guide.** The image was built and tested locally (Docker 29 + Colima, Apple-silicon/arm64, image size 2.3 GB): the container runs as UID 1000, loads the model, scores `sample_demo.csv` with the same probabilities as the notebooks, returns SHAP drivers, writes the prediction log, and returns 403 for `DELETE /api/logs` when `CHURN_ALLOW_LOG_CLEAR=0`. It has **not** yet been pushed to a real Space — Hugging Face builds for x86-64, so watch the first build log (step 5).

---

## What gets deployed

Only what the API needs at runtime (< 1 MB of project files + Python packages):

```
Dockerfile   requirements.txt   README.md (Space version, see step 3)
src/         app/               models/          data/sample_demo.csv
```
Notebooks, plots, the raw dataset, tests and the report stay in GitLab — they are not needed to serve predictions.

---

## Step 1 — (optional but recommended) test the image locally

Requires Docker (Docker Desktop, or `brew install docker colima && colima start` on macOS). From the project root:
```bash
docker build -t churn-dashboard .
```
```bash
docker run --rm -p 7860:7860 -e CHURN_ALLOW_LOG_CLEAR=0 churn-dashboard
```
Open http://localhost:7860 → **Try it with 50 sample customers** → click a customer (SHAP drivers load) → **Prediction log** tab shows 50 records. Stop with `Ctrl+C`.

---

## Step 2 — create the Space

1. Sign in at https://huggingface.co → **avatar → New Space**.
2. **Space name:** e.g. `churn-early-warning` · **License:** your choice (e.g. MIT).
3. **Select the Space SDK: Docker → Blank.**
4. **Hardware:** CPU basic (free) · **Visibility:** Public (or as your instructor requires).
5. **Create Space.**

---

## Step 3 — prepare the Space README

A Space is configured by a YAML header at the top of its `README.md`. The project README does not have one, so the Space uses the ready-made file `docs/hf_space_README.md`:

```yaml
---
title: Churn Early-Warning Dashboard
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```
`sdk: docker` tells Spaces to build the `Dockerfile`; `app_port: 7860` must match the port uvicorn listens on (`PORT=7860` in the Dockerfile).

---

## Step 4 — upload the files

Build a clean upload folder first so that only the runtime files (and the Space README) are sent:

```bash
rm -rf /tmp/churn-space && mkdir -p /tmp/churn-space/data && cp -R Dockerfile requirements.txt src app models /tmp/churn-space/ && cp data/sample_demo.csv /tmp/churn-space/data/ && cp docs/hf_space_README.md /tmp/churn-space/README.md && find /tmp/churn-space -name "__pycache__" -type d -prune -exec rm -rf {} +
```

### Option A — Hugging Face CLI (simplest; handles binary files automatically)

```bash
pip install -U huggingface_hub
```
```bash
hf auth login
```
(paste a token with **write** access from https://huggingface.co/settings/tokens — it is stored on your machine, never put it in a file or notebook)
```bash
hf upload <username>/<space-name> /tmp/churn-space . --repo-type=space --commit-message "Deploy churn dashboard"
```

### Option B — git

Hugging Face rejects pushes containing binary files unless they are tracked by Git LFS/Xet, and this project ships binaries (`.pth`, `.pkl`, `.npy`, `.png`, `.ico`):

```bash
git clone https://huggingface.co/spaces/<username>/<space-name> /tmp/space-repo && cd /tmp/space-repo
```
```bash
git lfs install && git lfs track "*.pth" "*.pkl" "*.npy" "*.npz" "*.png" "*.ico"
```
```bash
cp -R /tmp/churn-space/. . && git add -A && git commit -m "Deploy churn dashboard" && git push
```
(When git asks for a password, use your Hugging Face **write token**.)

### Option C — web upload
Space → **Files → Add file → Upload files**, drag the contents of `/tmp/churn-space`, commit.

---

## Step 5 — configure and watch the build

1. Space → **Settings → Variables and secrets → New variable** (public variables, not secrets):

| Variable | Value | Why |
|---|---|---|
| `CHURN_ALLOW_LOG_CLEAR` | `0` | Hides the **Clear log** button and makes `DELETE /api/logs` return 403, so visitors cannot wipe the log |
| `CHURN_LOG_BACKEND` | `jsonl` | Optional: skip the SQLite mirror on the Space's ephemeral disk |

2. Open the **Logs** tab (Build / Container). The first build takes about 5–10 minutes (it downloads PyTorch). Success looks like:
```
INFO churn-api: Model loaded: weighted_bce-6b911fc9 (τ* = 0.08)
INFO:     Uvicorn running on http://0.0.0.0:7860
```
3. The Space status turns **Running** and the dashboard appears in the **App** tab.

---

## Step 6 — verify the live deployment

Replace the host with your Space's direct URL (`https://<username>-<space-name>.hf.space`):

```bash
curl https://<username>-<space-name>.hf.space/api/health
```
Expect `"status":"ok","model_loaded":true`.

```bash
curl -F "file=@data/sample_demo.csv" https://<username>-<space-name>.hf.space/api/predict | head -c 400
```
In the browser: sample upload → table → customer panel with SHAP drivers → slider → **Prediction log** → **Model card** → `/docs`.

---

## Updating the deployment

After changing code or retraining (new `models/`), rebuild the upload folder (step 4) and upload again — the Space rebuilds automatically. The model version in the header chip (`weighted_bce-<hash>`) changes when the weights change, which is a quick way to confirm the new model is live.

---

## Things to know (and to say in the viva)

* **Storage is ephemeral.** On free hardware the container's disk is reset on every restart/rebuild, so `logs/predictions.jsonl` starts empty again. For real monitoring you would attach persistent storage (Space → Settings → Storage) and set `CHURN_LOG_DIR=/data/logs`, or ship logs to a database.
* **The Space sleeps** after ~48 hours without visitors on free hardware and wakes on the next visit (a 30–60 s cold start while the model loads).
* **It is public.** Anyone with the link can upload CSVs and call the API. Uploads are limited to 10 MB / 50,000 rows and are processed in memory, not stored — only the prediction log (customer ID, probability, tier, threshold) is written. Do not upload real customer data to a public Space.
* **Non-root user.** Spaces run containers as UID 1000; the Dockerfile creates that user and gives it ownership of the app folder so logging works.
* **No secrets needed.** The app calls no external services; never commit tokens.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build fails at `pip install` | Check the build log; usually a temporary network error — **Factory rebuild** in Settings |
| "Your push was rejected because it contains binary files" | Use Option A (`hf upload`) or track binaries with Git LFS (Option B) |
| Space stuck on "Starting", then error | `app_port` in the README header does not match the port (must be 7860) |
| Dashboard banner "Model not ready" (503) | `models/` was not uploaded completely — it needs `best_model.pth`, `scaler.pkl`, `encoder.pkl`, `feature_columns.pkl`, `model_config.json`, `shap_background.npy` |
| `PermissionError: logs/predictions.jsonl` or build fails at `mkdir logs` | Old Dockerfile — the current one creates `logs/` and `chown`s the app folder to the UID-1000 user |
| "Configuration error" on the Space page | YAML header missing or malformed at the very top of `README.md` |
| Customer panel shows "Could not compute drivers" | `shap_background.npy` missing, or the container ran out of memory (not expected on 16 GB) |
| Unpickling warning about scikit-learn versions | The image installed a different scikit-learn than the one used for training; pin it in `requirements.txt` (e.g. `scikit-learn==1.9.1`) and rebuild |

### Alternative: Render / Railway
The same `Dockerfile` works: create a *Web Service* from your GitLab repository, choose Docker, and set the environment variable `PORT` if the platform does not provide one. Add `CHURN_ALLOW_LOG_CLEAR=0` there too.
