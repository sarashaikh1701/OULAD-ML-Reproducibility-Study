This directory contains the full reproducibility materials for Paper 10: "A Withdrawal Prediction Model of At-Risk Learners" (David Tait, Stephen Lonn, Christopher Brooks, LAK 2019).

Purpose of the reproduction

The original study presents a withdrawal prediction framework for identifying at-risk students in OULAD courses. It compares multiple model families across balanced and unbalanced datasets, using both numeric and discretised indicator features.

Python implementation

This reproduction is implemented entirely in Python (P10.py / P10.ipynb) following the structure and methodology described in the paper. The workflow includes:

Data loading and filtering to the relevant course subset

Feature engineering for demographic, behavioural, and performance indicators

K-means discretisation of numeric indicators for certain model families

Definition of multiple feature set variants:

No indicators (demographics + raw VLE counts)

Numeric indicators

Discrete indicators

Balanced (SMOTE) vs unbalanced datasets

Model evaluation for: Decision Tree (J48 equivalent), Random Forest, TAN Bayesian Classifier, SVM, and MLP

Performance comparison tables across all variants

Any minor deviations from the paper (e.g., exact library versions, preprocessing defaults) are documented in the script header.

File list

P10.py – End-to-end reproduction script (entry-point) matching paper methodology

P10.ipynb – Jupyter Notebook version of the Python code for interactive exploration

requirements.txt – Pinned Python dependencies for the reproduction environment

A_Withdrawal_Prediction_Model_of_At-Risk_Learners_.pdf – Original paper