Credit Risk Prediction and Model Validation
Overview
A machine-learning project that analyses credit-card client data and predicts whether a client will default on their next month's payment.

Dataset
30,000 client records
23 predictor features used for modelling after excluding the client ID and target
Target: default payment next month
Class distribution: 77.9% non-default and 22.1% default
Methodology
Data quality checks and exploratory data analysis.
Analysis of repayment status and credit-limit groups.
Stratified training and test split.
Baseline and class-weighted logistic regression.
Balanced decision-tree classification.
Validation-based decision-threshold selection.
Final evaluation on an untouched test set.
Decision-tree feature-importance analysis.
Final Test Results
Accuracy: 77.10%
Precision: 48.45%
Recall: 55.46%
F1-score: 51.72%
ROC-AUC: 0.7592
Key Finding
Recent repayment status (PAY_0) was the dominant feature in the fitted decision tree, with a relative feature importance of approximately 74.97%.

Tools
Python, pandas, scikit-learn, Matplotlib, Google Colab.

Limitations
The model is evaluated on a historical dataset and is intended for educational and analytical purposes. Results do not establish suitability for real-world lending decisions. Feature importance indicates model reliance, not causation.

Files
Credit_Risk_Prediction.ipynb: analysis and modelling notebook.
credit_risk_model.pkl: trained preprocessing and classification pipeline.
model_config.json: selected threshold and recorded evaluation metrics

## Model Monitoring and Data Drift Detection

Extended the credit-risk analysis with a simulated monitoring workflow using the Population Stability Index (PSI) to compare reference training data with an incoming batch.

### Methodology
- Used the original 24,000-row training feature set as the reference dataset.
- Created a simulated incoming batch of 5,000 records.
- Introduced a controlled shift in `PAY_0` to test drift detection.
- Calculated PSI for all 23 input features and ranked them by drift score.

### Results
- `PAY_0` recorded a PSI of 2.0048 after the deliberate shift.
- All other features recorded PSI values of 0.0044 or lower.
- The monitoring chart highlights the feature with the largest distribution change.

![Feature drift monitoring](feature_drift_monitoring.png)

### Limitations
The incoming batch and feature shift are simulated for demonstration. Feature drift does not necessarily imply declining predictive performance. Evaluating model performance on incoming data requires reliable ground-truth outcomes.
