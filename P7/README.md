This directory contains the full reproducibility materials for Paper 7: "Joint RNN Models for Early Prediction of Student Performance in Online Learning" (He, Xiaoxiao; Tang, Jiliang; et al., 2020).

Purpose of having this Python script

Original study implementation
The original paper introduced a joint recurrent neural network (RNN) framework for predicting student performance early in online courses, using the Open University Learning Analytics Dataset (OULAD). The architecture jointly models weekly assessment scores and clickstream activity sequences, alongside demographic features, through parallel recurrent layers that are later concatenated for final prediction.

The authors evaluated both separate and joint RNN models, cycling training over multiple courses with periodic restarts of the computational graph to improve convergence. Metrics included accuracy, precision, and recall, tracked both overall and week-by-week.

Python reproduction
For consistency across the reproducibility project, the reproduction script (P7.py) follows the original architecture and training schedule closely, including:

Demographic one-hot encoding

Weekly aggregation of assessment and VLE clickstream data

Multi-input model assembly (demographics FCN, assessment RNN, clickstream RNN)

Joint GRU architecture with 7×256 hidden units per stream

Cyclic training (50-epoch cycles) with CSV logging and best-weight checkpoints

Per-course train/validation/test splits aligned with the paper’s schedule

Overall and week-by-week evaluation of accuracy, precision, and recall

Any departures due to environment constraints or API changes are noted in the script header.

Limitation in this reproduction

Running the full 250-epoch schedule for every course in the original sequence was not computationally efficient on the available system. For demonstration, runs beyond BBB_2014B were curtailed, meaning later course results are not included in the provided CSV and weight files.

File list

P7.py – Full Python reproduction of the paper’s methodology (entry-point script)

P7.ipynb – Jupyter Notebook version for interactive exploration

requirements.txt – Python dependencies for running the reproduction

AAA_2014J.csv / AAA_2014J_best.weights.h5 – Example course metrics log and saved model weights

BBB_2014B.csv / BBB_2014B_best.weights.h5 – Example course metrics log and saved model weights