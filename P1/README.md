This directory contains the full reproducibility materials for Paper 6: "Ouroboros: Early Identification of At-Risk Students Without Models Based on Legacy Data" (Martin Hlosta, Zdenek Zdrahal, Jaroslav Zendulka, LAK 2017)

Purpose of having this Python script

Original study implementation
The paper’s methodology was originally implemented in Python, using the Open University Learning Analytics Dataset (OULAD) to predict first-assignment (A1) submission using a self-learning approach that did not rely on historical course models. The original implementation trained models day-by-day (windowed) leading up to A1’s cut-off, using multiple classifiers (Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost) and evaluated primarily via PR-AUC, alongside Top-K precision/recall and feature importance inspection.

Python reproduction
For consistency across the reproducibility project and integration with the Python-based evaluation framework used for other papers, the reproduction (P1.py) retains the same overall structure and feature engineering steps described in the paper, including:

Demographic feature extraction

VLE clickstream aggregation and temporal statistics

Pre-presentation engagement metrics

60-day sliding activity window

Window-specific training and evaluation

Class balancing and model tuning for XGBoost

PR-AUC calculation, Top-K precision/recall, and feature importance reporting

Any intentional departures from the original pipeline (such as library version updates or API changes) are documented in the script header.

File list

P1.py – Full Python reproduction of the paper’s methodology

P1.ipynb – Jupyter Notebook version for interactive exploration

requirements.txt – Python dependencies for running the reproduction

Ouroboros.pdf – Original paper
