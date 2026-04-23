# Brain-Tumour-Survival-Analysis-Dashboard

## Overview
This project presents an interpretable, web-based machine learning system for predicting individual survival outcomes in patients with brain tumours. It is designed to support clinical decision-making, particularly in Sri Lanka, by combining accurate survival predictions with transparent, explainable insights.

The system enables clinicians to generate personalised prognoses, improve risk stratification, and support evidence-based treatment planning while maintaining high standards of interpretability and usability.

## Key Features

### Survival Models Implemented
- Cox Proportional Hazards (CPH)
- Aalen Additive Regression (AAR)
- Accelerated Failure Time (AFT)
- Random Survival Forest (RSF)
- Explainable Boosting Machines (EBM)

### Final Deployed Model
- **Accelerated Failure Time (AFT)**
- Test C-index: ~0.82–0.83
- Selected for interpretability and stable calibration
- Provides clinically meaningful time-ratio outputs

### Explainability
- SHAP (feature-level explanations)
- Global AFT time-ratio interpretation
- Insights into factors such as:
  - Age
  - Tumour grade
  - Surgery
  - Radiotherapy

### Risk Stratification
- Kaplan–Meier survival curves
- Risk groups:
  - Low
  - Medium
  - High
- Median survival threshold: ~37 months

### Patient Subgroup Discovery
- K-Means clustering (k=5)
- Identification of clinically meaningful patient archetypes
- Reflects tumour heterogeneity

### Web Application
- Real-time prediction interface
- Integrated explainability visualisations
- Secure and user-friendly design

## Dataset
- ~1,400 brain tumour patients
- Includes demographic, clinical, and treatment variables
- Has not been included

## Evaluation Metrics
- Concordance Index (C-index)
- Brier Score (6, 12, 36 months)
- Cross-validation
