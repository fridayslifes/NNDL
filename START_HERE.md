# Start here

Everything is already trained — you only need Python 3.10+ and an internet connection for the first run.

## Run the dashboard (one step)

| Your computer | Do this |
|---|---|
| **Windows** | Double-click **`run_app.bat`** |
| **macOS / Linux** | Open a terminal in this folder and run `./run_app.sh` (if it says "permission denied": `bash run_app.sh`) |

The first run creates a private Python environment in `.venv/` and installs the packages (a few minutes — PyTorch is large). After that it starts in seconds and opens **http://localhost:8000**.
Click **Try it with 50 sample customers**. Stop the app with `Ctrl+C`.

Port 8000 busy? macOS/Linux: `PORT=8010 ./run_app.sh` · Windows: `set PORT=8010` then `run_app.bat`.

## Re-run the notebooks (optional — reproduces every result)

After the first `run_app` (so `.venv/` exists):

* macOS / Linux: `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-7]_*.ipynb`
* Windows: `.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_eda.ipynb` (repeat for 02 … 07, in order)
* or open the notebooks in VS Code / Jupyter / Colab and run them in order 01 → 07.

## Run the tests (optional)

* macOS / Linux: `.venv/bin/python -m pytest -q` · Windows: `.venv\Scripts\python -m pytest -q` → 49 passed

## Where to read next

| File | What it is |
|---|---|
| `README.md` | Full documentation: results, structure, API, dashboard, configuration |
| `report/report.md` | Report draft (rewrite in your own words, export to PDF) |
| `report/viva_guide.md` | ~40 viva questions with answers |
| `docs/GITHUB_COLLABORATION_GUIDE.md` | How each member commits their own part on GitHub |
| `docs/GITLAB_COMMIT_GUIDE.md` | Alternative commit guide for GitLab |
| `docs/DEPLOY_HUGGINGFACE.md` | Docker deployment to Hugging Face Spaces |
