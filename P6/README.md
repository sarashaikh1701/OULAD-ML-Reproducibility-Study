This directory contains the full reproducibility materials for Paper 6: "Predicting academic performance of students from VLE big data using deep learning models" (Md Shoaib Ahmed, Shazia Sadiq, Ashad Kabir, Shazia W. Sadiq, Rolf A. Schwitter, 2021).

Purpose of having both `.py` and `.ipynb` scripts

Original study implementation 
The paper’s methodology was originally implemented using Python with TensorFlow 1.x and Keras 2.x, applied to the OULAD dataset for predicting student performance. The approach included preprocessing, feature engineering, and training deep learning models for binary classification tasks, with evaluation metrics such as accuracy, precision, recall, and AUC.

Python reproduction
For consistency across the reproducibility project and to integrate with the Python-based evaluation framework used for other papers, the original methodology was reproduced in Python (P6.py / P6.ipynb) with a pinned environment matching the original stack. This reproduction follows the paper’s structure but includes some documented deviations in the script header, such as:

- Explicit RNG seed setting for reproducibility  
- Training/validation/test split adapted to the reproducibility framework  
- CSV logging of metrics for each run  
- Saving trained Keras model weights to `.h5` for re-use  

File list

- `P6.py` – End-to-end Python reproduction script (entry-point)  
- `P6.ipynb` – Jupyter Notebook version for interactive exploration  
- `requirements.txt` – Python dependencies (pinned versions) for the reproduction  
- `logs/` – CSV logs of training/validation metrics for each model run  
- `weights/` – Saved Keras model weights (`.h5`)  
- `Predicting academic performance of students from VLE big data using deep.pdf` – Original paper  

