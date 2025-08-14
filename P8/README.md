This directory contains the full reproducibility materials for Paper 8:
"Ouroboros: Early identification of at-risk students without models based on legacy data"
(Miroslav Hlosta, Zdenek Zdrahal, Jaroslav Zendulka)

Original study implementation
The paper’s methodology was implemented in Python using scikit-learn (Logistic Regression, SVM, Random Forest, Naive Bayes) and XGBoost. Features were derived from student demographics, registration records, Virtual Learning Environment (VLE) activity, and pre-presentation engagement. The evaluation involved calculating Precision–Recall AUC (PR-AUC) for each weekly prediction window before the first assessment (A1) cut-off, as well as Top-K precision and recall metrics and feature importance rankings.

Python reproduction
For integration with the broader Python-based evaluation framework used for other papers in this reproducibility project, the pipeline was re-implemented in Python (P8.py / P8.ipynb). The reproduction follows the original structure but with minor departures (documented in the script header), such as:

Restriction to modules BBB, DDD, EEE, and FFF in presentation 2014J

A1 limited to TMA-type assessments with non-zero weight

Deterministic train_test_split via random_state=42

Skipping PR-AUC for windows with no positive labels

Per-window scale_pos_weight tuning for XGBoost to handle class imbalance

File list

P8.py – Full Python reproduction of the Ouroboros pipeline

P8.ipynb – Jupyter Notebook version of the reproduction for interactive exploration

requirements.txt – Pinned Python dependencies for reproducibility

Step2_CleanedDS_v1.csv – Intermediate dataset after cleaning and preprocessing

Step3_FeatureEngineered_v1.csv – Intermediate dataset after feature engineering

A Machine Learning Based Approach for Student Performance Evaluation in Educational Data Mining.pdf – Original paper
