This directory contains the full reproducibility materials for Paper 5: "Predicting At-Risk University Students in a Virtual Learning Environment via a Machine Learning Algorithm" (Kwok Tai Chui, Dennis Chun Lok Fung, Miltiadis D. Lytras, Tin Miu Lam, Computers in Human Behavior, 2020)

Purpose of having this Python script

Original study implementation
The paper’s methodology was originally implemented using the Open University Learning Analytics Dataset (OULAD) to predict student risk of failure based on final scores. The original implementation trained a Reduced Training Vector Support Vector Machine (RTV-SVM) per course module using demographic, registration, and behavioural features. Two tiers of data reduction were applied to the training set:

Tier 1 – Gaussian log-pdf–based filtering.

Tier 2 – Projection-based pruning.

The model was evaluated under four scenarios (S1–S4) using five-fold stratified cross-validation, and results were reported for binary (fail vs pass) and combined (fail + marginal vs pass) groupings. Module-level results were aggregated in Tables 3 and 5, with Table 4 reporting the class–gender distributions.

Python reproduction
For consistency across the reproducibility project and integration with the Python-based evaluation framework used for other papers, the reproduction (P5.py) retains the same overall structure and feature engineering steps described in the paper, including:

Demographic feature extraction.

VLE clickstream aggregation and assessment statistics.

Fixed category level definitions to ensure reproducible one-hot encoding.

Mean-imputation + z-score scaling for numeric predictors, k−1 one-hot encoding for categorical predictors.

Exact Tier 1 and Tier 2 RTV-SVM data reduction logic for S1–S4 scenarios.

Five-fold stratified cross-validation per module.

Aggregated results for Tables 3, 4, and 5 as reported in the paper.

Any intentional departures from the original pipeline (such as library version updates or API changes) are documented in the script header.

File list

P5.py – Full Python reproduction of the paper’s methodology
P5.ipynb – Jupyter Notebook version for interactive exploration
requirements.txt – Python dependencies for running the reproduction
Predicting at-risk university students in a virtual learning environment via a machine learning algorithm.pdf – Original paper
Table_3_Differences__Computed_-Original.csv – Difference report for reproduced vs original Table 3 values
Table_4_Differences__Computed_-Original.csv – Difference report for reproduced vs original Table 4 values
Table_5_Differences__Computed_-Original.csv – Difference report for reproduced vs original Table 5 values