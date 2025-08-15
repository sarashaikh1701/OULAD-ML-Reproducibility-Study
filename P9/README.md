This directory contains the full reproducibility materials for Paper 9:
"How Well a Student Performed: A Machine Learning Approach to Classify Students’ Performance on Virtual Learning Environment"
(Faisal M. Alnassar, Eesa Alshraideh, Eman Alnassar, 2021)

Purpose of this implementation

The original study used the Open University Learning Analytics Dataset (OULAD) to build binary classification models predicting whether a student will pass or fail based on demographic (D), engagement (E), and past performance (P) features. The models evaluated were k-Nearest Neighbors (k-NN), Support Vector Classifier (SVC), and Artificial Neural Network (ANN).

This Python-based reproduction (P9.py / P9.ipynb) implements the full end-to-end pipeline described in the paper:

Step 2 – Data loading, cleaning, and merging:

Filtering the target to Pass (1) / Fail (0) following the paper’s class definition.

Constructing feature blocks for Demographic, Engagement, and Past Performance.

Replacing all missing values with zero, per the paper’s approach.

Step 3 – Feature-set definition:

Creating the D, E, P sets and their combinations (D+E, D+P, E+P, D+E+P).

Standardizing numeric features and one-hot encoding categorical features.

Applying SelectKBest (f_classif) feature selection with k=10 features, per the paper’s method.

Step 4 – Model training and evaluation:

Fixed 80:20 train-test split, stratified on the label.

Running k-NN, SVC, and ANN with the paper’s stated hyperparameters.

Reporting Accuracy, F1-score, and Jaccard Index (J-Index).

File list

P9.py – Entry-point Python script implementing the full pipeline.

P9.ipynb – Jupyter Notebook version for interactive exploration.

merged_oulad_step2.csv / oulad_step2_preprocessed.csv / merged_oulad_paper3.csv – Intermediate preprocessed datasets.

knn_model.pkl – Saved k-NN model object for reuse.

preprocessing_pipeline.pkl – Saved preprocessing pipeline object (scaling, encoding, feature selection).

requirements.txt – Python dependencies pinned to the versions used in reproduction.