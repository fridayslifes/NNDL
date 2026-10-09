# Free Public Deployment on Render

Render provides **100% free hosting** for web services and connects directly to your GitHub repository ([fridayslifes/NNDL](https://github.com/fridayslifes/NNDL)).

> **Course Brief Reference:**
> *"Deploy publicly (Hugging Face Spaces, Render or similar) or demo locally with a recorded video."*
> Render is explicitly recommended in the course guidelines!

---

## 2-Minute Deployment Steps (Free, No Card Required)

### 1. Sign In to Render with GitHub
1. Go to **[https://dashboard.render.com](https://dashboard.render.com)**.
2. Click **Sign in with GitHub** (log in as `fridayslifes`).

---

### 2. Create New Web Service
1. In the top right, click **New +** → select **Web Service**.
2. Select **Build and deploy from a Git repository**.
3. Under **Connect a repository**, find and click **Connect** next to **`fridayslifes/NNDL`**.

---

### 3. Configure and Launch
Render will auto-detect the repository settings:
* **Name:** `telco-churn-dashboard`
* **Region:** Any (e.g., Singapore / Oregon / Frankfurt)
* **Branch:** `main`
* **Runtime:** `Python 3`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
* **Instance Type:** **Free**

Click **Create Web Service** at the bottom!

---

## What Happens Next?
Render will automatically:
1. Clone your GitHub repository.
2. Install dependencies (`requirements.txt`).
3. Start the FastAPI server on their public cloud.
4. Provide a free, permanent public URL:  
   **`https://telco-churn-dashboard-xxxx.onrender.com`**

You can paste this public URL directly into your project submission or viva presentation!
