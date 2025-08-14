This directory contains the full reproducibility materials for Paper 4: "Predicting At-Risk Students Using Clickstream Data in the Virtual Learning Environment" (Naif Radi Aljohani, Ayman Fayoumi, Saeed-Ul Hassan, Sustainability 2019)

Purpose of having this Python script

Original study implementation
The paper’s methodology was originally implemented using the Open University Learning Analytics Dataset (OULAD) to predict student final results (Pass/Fail) from their week-by-week clickstream activity in the Virtual Learning Environment (VLE). The original study converted VLE logs into sequential, weekly 20-activity vectors per student, trained a deep Long Short-Term Memory (LSTM) model to make early predictions at multiple week cut-offs, and compared its performance to baseline models (Logistic Regression, SVM, Artificial Neural Network). Metrics reported included accuracy, precision, and recall per week.

Python reproduction
For consistency across the reproducibility project and integration with the Python-based evaluation framework used for other papers, the reproduction (P4.py) implements the same feature engineering, model training, and evaluation steps described in the paper, including:

Loading and preprocessing OULAD tables (studentInfo, studentVle, vle, courses)

Merging Distinction into Pass and filtering out Withdrawn students

Generating week-by-week clickstream counts for 20 VLE activity types

Padding sequences to 38 weeks with masking for LSTM input

Training deep LSTM models with 3 layers (100–200–300 units) and dropout, at cut-offs: Week 5, Week 10, Week 20, and Week 38

Saving models, predictions, and training histories

Evaluating predictions with accuracy, precision, and recall

Implementing baseline models (Logistic Regression, SVM, MLP) on aggregated week-wise data for comparison

Any intentional departures from the original pipeline (e.g., handling of repeated students, exact ANN baseline architecture, random seed handling) are noted in the script header.

File list

P4.py – Full Python reproduction of the paper’s methodology

P4.ipynb – Jupyter Notebook version for interactive exploration

requirements.txt – Python dependencies for running the reproduction

Predicting_At-Risk_Students_Using_Clickstream_Data.pdf – Original paper

history_logs/ – Saved Keras training histories (.npy)

models/ – Saved trained LSTM models (.h5)

predictions/ – Saved prediction arrays for each week cut-off (.npy)

lstm_weekwise_results.csv – LSTM evaluation results

baseline_model_results.csv – Baseline model evaluation results

Comparison Graph.py – Script for visualizing LSTM vs. baseline performance

output.png – Example output graph of results