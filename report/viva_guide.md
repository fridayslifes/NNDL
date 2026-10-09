# Viva Question Bank — Telecom Churn Early-Warning System

Answers use the numbers produced by this repository. Both Riya and Praveen should be able to answer **every** question; the tag only shows who is most likely to be asked first. Where our results differ from the "textbook" answer in the project brief, the answer below says so — examiners reward precision over slogans.

---

## Brief questions (with corrections where the data disagree)

**Q1 (Riya) — Why did you fit the scaler only on the training set?**
Validation and test stand in for unseen future customers. Fitting `StandardScaler`/`OneHotEncoder` on them would leak their mean, variance and category list into training and bias our metrics upwards. `fit_preprocessors` receives only the training frame; everything else — validation, test, uploaded CSVs — goes through `.transform()`. Evidence in notebook 02: scaled training columns have mean exactly 0 and std 1, validation/test do not.

**Q2 (Riya) — Why PR-AUC over ROC-AUC?**
ROC-AUC uses the false-positive *rate* FP/(FP+TN); with 2.77 non-churners per churner the large TN count keeps that rate small, so ROC-AUC looks flattering (ours ≈ 0.84). PR-AUC uses precision TP/(TP+FP), which directly exposes wasted offers, and its random baseline is the positive rate (0.265) rather than 0.5. Our PR-AUC of 0.63 is 2.4× random — a more honest picture of minority-class performance.

**Q3 (Praveen) — How does the threshold change if the offer cost rises?**
It rises. Send an offer iff p·C_lost > C_offer, i.e. τ = C_offer / C_lost. In notebook 06, raising C_offer from ₹1,500 to ₹6,000 (ratio 0.40) moved τ\* from 0.08 to 0.44 and shrank the campaign from 975 to 354 validation customers — the empirical τ\* tracks the theoretical line because our probabilities are calibrated. You can demonstrate it live by editing C_offer in the dashboard and clicking the **Cheapest here** preset (or clicking the cost curve).

**Q4 (Praveen) — Where are predictions logged and why does it matter?**
`logs/predictions.jsonl` (append-only JSON Lines) plus an SQLite mirror `logs/predictions_log.db`, one record per customer: timestamp, customer_id, churn_probability, risk_tier, threshold_used, flagged_for_offer, batch_id, source, model_version. It enables drift monitoring (score distribution shifting over time), auditing (which model and threshold produced a decision — the version string contains a SHA-256 prefix of the weights), and measuring real campaign impact once outcomes are joined back.

**Q5 (Both) — How does focal loss differ from BCE?**
BCE = −log p_t. Focal loss = −α_t (1 − p_t)^γ log p_t. The modulating factor (1 − p_t)^γ down-weights **easy, well-classified examples of either class** (not only negatives): at p_t = 0.95 and γ = 2 the loss is cut 400×, while a hard example (p_t = 0.3) keeps about half its loss. The class imbalance itself is addressed by α_t (0.75 for churners vs 0.25 for non-churners). With γ = 0, α = 0.5 it equals ½·BCE — we unit-test this. We implemented it from logits with `logsigmoid` for numerical stability and did not use `torchvision.ops.sigmoid_focal_loss`.

**Q6 (Both) — How did Adam compare with SGD-momentum and RMSprop?** *(correct the brief's claim with our data)*
Same initial weights and batch order, 30 epochs. Adam had the **highest** loss after epoch 1 (0.604 vs 0.515 SGD, 0.494 RMSprop): its early steps are ≈ η = 0.001 per parameter because m̂/√v̂ ≈ ±1, while SGD-momentum's effective step is η/(1−β) = 0.1. By epoch 5 all three were within 0.005; by epoch 30 Adam had the lowest training loss (0.391) and the best validation PR-AUC (0.629). So Adam did **not** converge dramatically faster here — the standardised inputs and BatchNorm make the loss surface well conditioned — but it reached the best solution without tuning, combining momentum (smooths noisy mini-batch gradients) with RMSprop's per-parameter scaling (bigger effective steps for rarely-active one-hot features). That robustness is why it is our default.

**Q7 (Both) — Why ReLU in hidden layers instead of sigmoid?**
σ′(z) = σ(z)(1 − σ(z)) ≤ 0.25 and ≈ 0 when |z| is large; back-propagation multiplies one such factor per layer, so gradients vanish geometrically (≤ 0.25¹⁰ ≈ 10⁻⁶ after ten layers). ReLU′ = 1 for active units, preserving gradient magnitude, and is cheap to compute. The **output** keeps a sigmoid because we need a probability in (0, 1); combined with BCE its derivative cancels, leaving the clean gradient ŷ − y.

---

## Data & preprocessing

**Why fill blank TotalCharges with 0 rather than the mean?** All 11 blanks have tenure = 0 and they are the only tenure-0 rows: brand-new, never-billed customers, so their true cumulative charge is 0. The mean (≈2,283) would invent billing history.

**Why stratify the split?** So each split has the same 26.5 % churn rate; otherwise validation/test metrics could shift simply because the class balance differs.

**How do you guarantee the test set was used once?** `load_splits()` only returns test arrays with `include_test=True`, used solely in notebook 07. Learning rate, imbalance strategy, calibration and τ\* were all chosen on validation. `sample_demo.csv` comes from test rows but carries no labels and is only displayed, never scored against ground truth.

**What happens if an uploaded CSV has a category never seen in training?** `OneHotEncoder(handle_unknown="ignore")` encodes it as all zeros, and `clean_dataframe` reports it as a warning in the UI. Rows with non-numeric/negative tenure or charges are skipped and listed; missing columns return HTTP 422 naming every missing column.

**Why `drop="if_binary"`?** A Yes/No column becomes one 0/1 feature instead of two perfectly collinear ones; multi-level columns keep every level.

## From-scratch perceptron

**Derive ∂L/∂W.** ∂L/∂ŷ = (ŷ − y)/(ŷ(1−ŷ))·(1/m); ∂ŷ/∂z = ŷ(1−ŷ) → ∂L/∂z = (ŷ − y)/m; z = XW + b → ∂L/∂W = Xᵀ(ŷ − y)/m, ∂L/∂b = mean(ŷ − y).

**How do you know it's correct?** Central finite differences [L(θ+ε) − L(θ−ε)]/2ε for all 41 parameters agree with the analytic gradient to a relative error of 2.4 × 10⁻¹⁰, and the trained model reaches the same loss (0.409 vs 0.408) and PR-AUC (0.640 vs 0.642) as scikit-learn.

**Why is it basically logistic regression?** A single sigmoid unit trained on BCE *is* logistic regression; only the optimiser differs (our mini-batch GD vs L-BFGS + small L2). Rosenblatt's perceptron used a step function and the perceptron rule, which is not differentiable and gives no probabilities.

**Why do the perceptron and LogReg weights differ (r = 0.88) if predictions match?** Collinearity: TotalCharges ≈ tenure × MonthlyCharges, so many weight combinations give the same predictions; L2 and a different optimiser settle on different ones. Individual linear coefficients are unreliable importance measures here — hence SHAP.

**Why did η = 0.5 behave badly?** Mini-batch gradients are noisy estimates; the parameter jitter scales with η, so the loss kept bouncing (std 0.17 over the last 50 epochs) instead of settling.

## MLP & training

**Why BatchNorm?** Normalises each hidden unit's pre-activation over the mini-batch, then rescales with learned γ, β: steadier activation distributions, tolerance to larger learning rates, mild regularisation from batch noise. In `eval()` it uses running averages, so one customer can be scored deterministically.

**Why `drop_last=True`?** 4,225 = 66 × 64 + 1. A one-sample batch has zero variance, and PyTorch raises "Expected more than 1 value per channel when training". Reshuffling each epoch means a different row is dropped each time.

**How does dropout regularise, and why is validation loss sometimes below training loss?** Each hidden unit is zeroed with p = 0.3 during training (survivors scaled by 1/0.7), preventing co-adaptation — an implicit ensemble. Training loss is measured with this handicap on; validation loss uses the full network.

**What exactly does `weight_decay=1e-4` do in Adam?** Adds λθ to each gradient, i.e. an L2 penalty λ/2‖θ‖². (Nuance: in Adam this penalty is then rescaled by the adaptive denominator; AdamW decouples the decay from the adaptive step. We used the brief's Adam + weight_decay.)

**What is early stopping monitoring and why val loss rather than PR-AUC?** The brief specifies validation loss; it is smooth and directly what the model optimises. We keep the lowest-val-loss weights and stop after 10 non-improving epochs. PR-AUC is logged each epoch (right panel of the learning-curve figure) for reference.

**Why BCELoss for plain training but BCEWithLogitsLoss for weighted?** `BCEWithLogitsLoss` fuses sigmoid + BCE using the log-sum-exp trick (numerically stable) and supports `pos_weight`. The model exposes both `logits()` and `forward()` (probabilities) so each loss gets what it expects.

## Imbalance, calibration & threshold

**What is `pos_weight`?** N_neg/N_pos = 3,104/1,121 = 2.77: every churner's loss term is multiplied by 2.77 so both classes contribute equally to the total loss.

**Why oversample only the training split?** Validation/test must keep the real 26.5 % mix or their metrics would describe a population that does not exist; duplicates in evaluation data would also double-count customers.

**Why did you pick weighted BCE if the differences are tiny?** The rule, fixed before looking at results, was the highest mean validation PR-AUC over seeds 42–46. Weighted BCE won on seed 42 and on average (0.6381 ± 0.0025), with the lowest variance. We state openly that all four are within ≈0.01 — the main effect of imbalance handling is to shift the operating point, which we set by cost anyway.

**Why calibrate, and does it change the ranking?** Weighted BCE inflates probabilities (mean predicted 0.43 vs true 0.265). The dashboard's tiers and ₹ projections assume p is a real probability. Platt scaling σ(a·z + b) (a = 1.012, b = −1.088) fixes the scale (Brier 0.175 → 0.139). Because a > 0 it is monotonic: PR-AUC, ROC-AUC and every achievable confusion matrix are unchanged.

**Why is τ\* so low (0.08) — isn't flagging 69 % of customers crazy?** Missing a churner costs ₹15,000; an unnecessary offer costs ₹1,500. Anyone with more than a 10 % churn risk has a positive expected value for an offer (p·15,000 − 1,500 > 0). The result: 97 % of test churners caught, cost ₹16.5 L vs ₹31.2 L at τ = 0.5. If offers were costlier or less effective, τ\* would rise (Q3).

**F1 at τ\* is only 0.537 — did you fail the F1 ≥ 0.60 criterion?** At the F1-optimal threshold chosen on validation (0.33) test F1 is 0.613, so the model can meet it. At τ\* F1 is lower by design: F1 treats a false alarm and a missed churner as equally bad, while the business says a miss costs 10×. We report both.

**The MLP didn't beat logistic regression — why, and is that a failure?** Test PR-AUC 0.632 vs 0.634; paired-bootstrap 95 % CI of the difference [−0.010, +0.005] includes 0 → statistically tied ("matches"). The strongest signals act almost additively on log-odds (EDA), and 4,225 training rows limit what extra capacity can learn. The project's biggest gain comes from cost-based thresholding + calibration, not from the architecture.

**What assumption does the cost formula make?** That an offer always retains a true churner. With acceptance rate r, a TP costs C_offer + (1 − r)·C_lost; the optimal threshold rises to C_offer / (r·C_lost).

## Explainability

**How does DeepExplainer work?** A DeepLIFT-style method: it back-propagates "contribution scores" relative to a background set (200 training customers) through each layer (Linear, BatchNorm, ReLU, Sigmoid), approximating Shapley values. Contributions add up exactly: prediction = base value + Σφ (our additivity error ≈ 10⁻⁷).

**Why sum SHAP values across one-hot columns?** SHAP values are additive, so the total effect of "Contract" is the sum of its three dummy columns' effects — a human-readable driver ("Contract = Month-to-month: +6.8 pp").

**Does a large SHAP value mean changing that feature will change churn?** No — SHAP explains the model's reasoning, not causality. It points to plausible levers (contract type, tech support) that should be tested in a real campaign.

## Application

**Walk through what happens when a CSV is dropped on the page.** JS sends a multipart POST to `/api/predict` with the current τ. FastAPI validates the file (extension, size, not empty), parses it (UTF-8/Latin-1), runs `ChurnPredictor.predict_dataframe` in a worker thread — same `clean_dataframe` → scaler/encoder `.transform` → MLP logits → Platt → tier as in training — sorts by probability, logs one record per customer, and returns JSON. The browser renders the table and computes campaign economics locally.

**Why compute the slider economics in the browser?** Instant feedback with no network round-trip. The formulas are identical to `cost_optimizer.expected_cost_unlabeled`: campaign = #(p ≥ τ)·C_offer; missed = Σ_{p<τ} p·C_lost; do-nothing = Σ p·C_lost; savings = do-nothing − (campaign + missed).

**Why `run_in_threadpool`?** Pandas, PyTorch and SHAP are CPU-bound and synchronous; running them directly in an `async` endpoint would block the event loop and freeze every other request.

**How do you prevent the dashboard from being broken by a malicious CSV?** Every value from the file is HTML-escaped before insertion into the page; uploads are size- and row-limited; Pydantic validates the JSON endpoint (unknown categories → 422).

**What if the model files are missing?** The server still starts; `/api/health` reports `model_not_ready`, prediction endpoints return 503 with the reason, and the UI shows a banner telling the user to run the notebooks.
