"""
Telco Customer Churn Early-Warning System — reusable source package.

Every notebook in ``notebooks/`` and the FastAPI backend in ``app/`` import from
this package so that exactly the same preprocessing, model definition, loss
functions and cost logic are used during training *and* at inference time.
Keeping one implementation of each step is the simplest defence against
training/serving skew (a classic production ML bug in which the web app
transforms data slightly differently from the training notebook).

Modules
-------
config              Paths, random seeds, hyper-parameters, business costs.
data_preprocessing  Cleaning, stratified splitting, leakage-free encoding/scaling.
numpy_perceptron    From-scratch single-layer perceptron (pure NumPy).
model               PyTorch MLP (BatchNorm + Dropout) and Platt-calibrated wrapper.
focal_loss          From-scratch focal loss in PyTorch.
training            Training loop, early stopping, imbalance strategies.
metrics             Precision / recall / F1 / PR-AUC / ROC-AUC helpers.
calibration         Platt scaling so probabilities can be read as real risks.
cost_optimizer      Business-cost threshold sweep and savings calculations.
explainability      SHAP DeepExplainer helpers and per-customer top drivers.
inference           ``ChurnPredictor`` — the single object the web app talks to.
plotting            Shared, colour-blind-safe matplotlib styling.
"""

__version__ = "1.0.0"
