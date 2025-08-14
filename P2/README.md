This directory contains the full reproducibility materials for Paper 4:
"The Application of Gaussian Mixture Models for the Identification of At-Risk Learners in Massive Open Online Courses"
(Raghad Alshabandar, Abir Jaafar Hussain, Robert Keight, Andy Laws, Thar Baker)

Purpose of having both R and Python scripts

Original study implementation
The paper’s methodology was originally implemented in R, using packages such as mclust, caret, and pROC.
The file P2.R is a paper-faithful reproduction of the original R pipeline, matching the preprocessing, feature engineering, rare-feature pruning, log-transformations, class balancing, and threshold optimization exactly.

Python reimplementation
For consistency across the reproducibility project and to integrate with the Python-based evaluation framework used for other papers, the R pipeline was re-implemented in Python (P2.py / P2.ipynb).
The Python version follows the same overall structure but has some intentional departures from the original R code (documented in the script header), such as:

No rare-feature pruning

Raw counts instead of log-transformed features

No class upsampling

Fixed 0.5 probability threshold

Additional baseline models (Logistic Regression, kNN) alongside EDDA

File list

P2.R – Full paper-faithful R reproduction

P2.py – Python reimplementation for integration with the broader reproducibility framework

P2.ipynb – Jupyter Notebook version of the Python code for interactive exploration

requirements.txt – Python dependencies for the Python version

all_metrics.csv – Detailed per-run metrics

Full_Model_Performance_Comparison.csv – Aggregated metrics table

results.PNG – Screenshot of final results

Application of Gaussian mixture model.pdf – Original paper
