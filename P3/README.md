This directory contains the full reproducibility materials for Paper 3: "Student Success Prediction and Trade-off Between Prediction Accuracy and Interpretability" (source: Open University Learning Analytics Dataset study)

Purpose of having this Python script

Original study implementation
The paper’s methodology was implemented using the Open University Learning Analytics Dataset (OULAD) to predict student final outcomes (Pass/Fail) based on demographic data and virtual learning environment (VLE) activity. The original study examined multiple feature sets: demographic only, activity only, and combinations, and compared predictive performance across several supervised learning algorithms. Additionally, the study explored the interpretability trade-off by also performing unsupervised clustering on student activity patterns.

Python reproduction
For consistency across the reproducibility project and integration with the unified Python-based evaluation framework, the reproduction (P3_entry.py) maintains the same preprocessing, feature engineering, and model evaluation steps described in the paper, including:

Filtering of students (removal of “Withdrawn” outcomes and banked assessments)

Binary encoding of final outcomes (Pass + Distinction = 1, Fail = 0)

Construction of 245-day activity vectors (count, binary, and normalized forms)

Demographic feature extraction with one-hot encoding of categorical attributes

Multiple feature set combinations for model comparison

5-fold stratified cross-validation for model evaluation using Accuracy, Precision, Recall, and F1-score

Supervised models: Decision Tree, Random Forest, Logistic Regression, and SVM (RBF kernel)

Unsupervised analysis: K-Means clustering (k=9) on binary activity vectors, with visualization of average interaction curves

Any intentional departures from the original pipeline (such as fixing library versions for compatibility with the reproducibility framework) are documented in the script header.

File list

P3_entry.py – Full Python reproduction of the paper’s methodology in a single end-to-end script

P3.ipynb – Jupyter Notebook version for interactive exploration

requirements.txt – Python dependencies for running the reproduction

Student success prediction and trade off.pdf – Original paper
