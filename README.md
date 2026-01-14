\# Author: Owen Nda Diche

\# Energy Churn Prediction \& Price Sensitivity (Random Forest)



\## Overview

This project predicts customer churn for an energy retailer and investigates whether price-related behavior is associated with churn risk.  

It follows an end-to-end workflow: \*\*EDA → Feature Engineering → Modeling → Threshold Decisioning → Executive Summary\*\*.



> Note: Raw datasets are not included in this repo.



---



\## Business Problem

Customer churn impacts revenue. The client hypothesis was that churn is driven by \*\*price sensitivity\*\* (customers switching to cheaper providers).  

Goal: build a churn model and quantify key churn drivers, especially price-based features.



---



\## Data

Two datasets were used:

\- \*\*Client data\*\*: customer attributes, consumption, margin, tenure, churn label

\- \*\*Pricing data\*\*: historical prices over time per customer



\*\*EDA highlights\*\*

\- Churn is imbalanced: \*\*~9.72% churners\*\* (1) vs \*\*~90.28% non-churners\*\* (0)

\- Pricing covers \*\*100%\*\* of client IDs (merge-ready)

\- Heavy skew/outliers exist in consumption and margin fields (typical for energy usage)



---



\## Approach

\### 1) Exploratory Data Analysis

\- Validated date columns and data types

\- Missingness checks (notably `channel\_sales` missing ~25.5%)

\- Distribution checks for skew/outliers

\- Churn imbalance quantified



\### 2) Feature Engineering

\- Customer lifecycle / tenure features (e.g., months active, product modification counts)

\- Consumption features (12m totals, last month usage)

\- Pricing features aggregated per customer (e.g., spreads, variability, monthly differences)



\### 3) Modeling (Random Forest)

\- Preprocessing via pipeline: imputation + one-hot encoding

\- RandomForestClassifier with `class\_weight="balanced"`

\- Metrics used: \*\*ROC-AUC\*\*, Precision/Recall/F1, Confusion Matrix

\- Added a \*\*baseline\*\* (predict all non-churn) to show why accuracy is misleading



\### 4) Threshold Tuning (Decisioning)

Default threshold (0.5) was too conservative for churn detection, so we tuned the threshold to maximize F1 and improve recall.



---



\## Results (Test Set)

\### Baseline (predict all non-churn)

\- Accuracy: \*\*0.9028\*\*

\- Recall (churn): \*\*0.0000\*\*

\- F1: \*\*0.0000\*\*



\### Random Forest @ default threshold (0.5)

\- ROC-AUC: \*\*0.705\*\*

\- Accuracy: \*\*0.9100\*\*

\- Precision: \*\*0.8889\*\*

\- Recall: \*\*0.0845\*\*

\- F1: \*\*0.1543\*\*



\### Tuned threshold (Max F1)

\- Best threshold: \*\*0.194\*\*

\- Accuracy: \*\*0.8682\*\*

\- Precision: \*\*0.3355\*\*

\- Recall: \*\*0.3627\*\*

\- F1: \*\*0.3486\*\*



\*\*Key takeaway:\*\* tuning the threshold increased churn capture from \*\*8.5% → 36.3% recall\*\* (≈4× more churners identified).



---



\## Key Drivers (Feature Importance)

Top signals included:

\- Profit/value variables: `margin\_gross\_pow\_ele`, `margin\_net\_pow\_ele`, `net\_margin`

\- Consumption behavior: `cons\_12m`, `cons\_last\_month`, `imp\_cons`

\- Tenure/changes: `months\_activ`, `months\_modif\_prod`

\- Price sensitivity proxies: spreads/variability and month-to-month differences in price components



---



\## Recommendations

\- Use the model as a \*\*risk ranking tool\*\* (ROC-AUC 0.705) and choose a threshold based on business costs.

\- For retention campaigns, use the tuned threshold (\*\*0.194\*\*) as a balanced starting point.

\- Run an uplift / A/B retention test:

&nbsp; - Target high-risk customers flagged by the model

&nbsp; - Compare churn reduction vs incentive cost

\- Segment price-sensitive customers using price variability/spread features for targeted offers.



---



\## Project Artifacts

\- Notebooks: `notebooks/`

\- Figures: `figures/`

\- Executive summary slide: `reports/executive\_summary\_slide.pdf`



---



\## How to Run

```bash

pip install -r requirements.txt



