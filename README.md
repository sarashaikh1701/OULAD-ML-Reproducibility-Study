# OULAD-ML-Reproducibility-Study
Reproducing Machine Learning Models on OULAD: A Multi-Paper Study
Sara Shaikh
Supervisor: Dr. Emma Howard
University of Dublin, Trinity College, 2025

A dissertation submitted in partial fulfilment of the requirements for the degree of Master of Science in Computer Science (Data Science)

This repository contains the complete code and artefacts for reproducing results from ten published machine learning studies based on the Open University Learning Analytics Dataset (OULAD). The work forms part of a structured reproducibility audit conducted as the author’s MSc dissertation, covering both Experiment Reproducibility (R1) and Data Reproducibility (R2).

To download the dataset, use: https://analyse.kmi.open.ac.uk/open-dataset

The included scripts recreate the data processing, feature engineering, model training, and evaluation pipelines described in each paper, following a unified, version-controlled environment specification for transparency and repeatability. Each paper’s reproduction is contained in its own folder (P1–P10) with:

Entry-point Python script that runs end-to-end: data loading, preprocessing, splitting, model training, and evaluation.

Requirements file pinning exact library versions.

Outputs (when applicable) such as trained models (.h5 or .joblib), evaluation metrics in CSV, and feature importance rankings.

By making the reproduction process transparent and repeatable, this work contributes to strengthening research reliability in the educational data mining community.

