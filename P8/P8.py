#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#P8 Reproduction Script
"""
Title: "A Machine Learning Based Approach for Student Performance Evaluation in Educational Data Mining"
Authors: Muhammad Sammy Ahmad, Ahmed H. Asad, Ammar Mohammed
Source: Proceedings of the 2021 International Mobile, Intelligent, and Ubiquitous Computing Conference (MIUCC)
DOI: 10.1109/MIUCC52538.2021.9447602

Purpose:
This script reproduces the methodology and results of the above paper using the 
Open University Learning Analytics Dataset (OULAD). It executes the complete 
pipeline described in the paper:
    - Data loading and preprocessing
    - Feature engineering
    - Model training (Artificial Neural Network, Random Forest)
    - Evaluation (accuracy, classification metrics, cross-validation)
"""


# In[ ]:


import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# In[ ]:


random.seed(42)
np.random.seed(42)
try:
    tf.random.set_seed(42)      # TF 2.x
except AttributeError:
    tf.set_random_seed(42)      # TF 1.x


# In[ ]:


# Step 2: Data Manipulation (using the exact v1.0 OULAD URLs)

# 1. Load the exact v1.0 CSVs from the GitHub mirror
url_info   = "https://raw.githubusercontent.com/marloft/MachineLearning/master/Documents/ML/PhD/Datasets/Open%20University%20Learning%20Analytics%20Dataset%20-%20OULAD/studentInfo.csv"
url_assess = "https://raw.githubusercontent.com/marloft/MachineLearning/master/Documents/ML/PhD/Datasets/Open%20University%20Learning%20Analytics%20Dataset%20-%20OULAD/studentAssessment.csv"

df_info   = pd.read_csv(url_info)
df_assess = pd.read_csv(url_assess)

# 2. Keep only Distinction/Pass/Fail and drop Ireland & North Region (as per paper)
df_info = df_info[df_info["final_result"].isin(["Distinction", "Pass", "Fail"])]
df_info = df_info[~df_info["region"].isin(["Ireland", "North Region"])]

# 3. Aggregate assessment features per student:
agg_dict = {
    "date_submitted": "mean",
    "is_banked":       "sum",
    "id_assessment":   "count",
    "score":           "mean"
}
if "num_of_prev_attempts" in df_assess.columns:
    agg_dict["num_of_prev_attempts"] = "sum"

df_assess_agg = (
    df_assess
    .groupby("id_student")
    .agg(agg_dict)
    .rename(columns={
        "id_assessment": "num_assessments",
        "score":          "avg_score"
    })
    .reset_index()
)

# 4. Merge back to df_info on id_student
df_merged = pd.merge(
    df_info,
    df_assess_agg,
    how="inner",
    on="id_student"
)

# 5. Drop any rows with NaNs
df_merged = df_merged.dropna()

# 6. Save the cleaned v1.0 merge for Step 3
df_merged.to_csv("Step2_CleanedDS_v1.csv", index=False)
print("Step 2 (v1.0) complete. Shape:", df_merged.shape)
print("Saved as Step2_CleanedDS_v1.csv")


# In[ ]:


# Step 3: Feature Engineering (using Step2_CleanedDS_v1.csv)

# 1. Load the v1.0 cleaned, merged dataset from Step 2
df = pd.read_csv("Step2_CleanedDS_v1.csv")

# 2. Drop "id_student"
df = df.drop(columns=["id_student"], errors="ignore")

# 3. Encode the target variable "final_result": Distinction→0, Fail→1, Pass→2
df["final_result"] = df["final_result"].map({
    "Distinction": 0,
    "Fail":        1,
    "Pass":        2
})

# 4. Label-encode binary features: gender (F→0, M→1), disability (N→0, Y→1)
df["gender"]     = df["gender"].map({"F": 0, "M": 1})
df["disability"] = df["disability"].map({"N": 0, "Y": 1})

# 5a. Ordinal-encode highest_education
edu_map = {
    "No Formal quals":        0,
    "Lower Than A Level":     1,
    "A Level or Equivalent":  2,
    "HE Qualification":       3,
    "Post Graduate Qualification": 4
}
df["highest_education"] = df["highest_education"].map(edu_map)

# 5b. Ordinal-encode age_band
age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
df["age_band"] = df["age_band"].map(age_map)

# 5c. Ordinal-encode imd_band
imd_map = {
    "0-10%":   0,
    "10-20%":  1,
    "20-30%":  2,
    "30-40%":  3,
    "40-50%":  4,
    "50-60%":  5,
    "60-70%":  6,
    "70-80%":  7,
    "80-90%":  8,
    "90-100%": 9
}
df["imd_band"] = df["imd_band"].map(imd_map)

# 6. Ensure "date_submitted" is numeric (v1.0 files already have it numeric)
if df["date_submitted"].dtype == object:
    df["date_submitted"] = pd.to_datetime(df["date_submitted"], errors="coerce")
    df["date_submitted"] = df["date_submitted"].astype(np.int64) // 10**9

# 7. One-hot encode region, code_module, code_presentation (drop_first=True)
df_featured = pd.get_dummies(
    df,
    columns=["region", "code_module", "code_presentation"],
    drop_first=True
)

# 8. Fill any remaining NaNs with 0
df_featured = df_featured.fillna(0)

# 9. Verify number of input features (excluding "final_result")
features = [c for c in df_featured.columns if c != "final_result"]
print("Number of feature columns (should be 31):", len(features))

# 10. Save the feature-engineered dataset for Step 4
df_featured.to_csv("Step3_FeatureEngineered_v1.csv", index=False)
print("Step 3 (v1.0) complete. Saved as Step3_FeatureEngineered_v1.csv")


# In[ ]:


#Step 4: ANN

# 1. Load the feature‐engineered dataset from Step 3
df = pd.read_csv("Step3_FeatureEngineered_v1.csv")

# 2. Separate X (features) and y (target)
#    – Drop “final_result” from X
X = df.drop(columns=["final_result"]).values   # 30 columns
y = df["final_result"].values                  # values in {0,1,2}

# 3. Normalize features (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into Train/Test (80/20), stratified by y
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. One‐hot encode the targets for Keras
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat  = to_categorical(y_test,  num_classes=3)

# 6. Build the ANN model function
#    – 30 inputs → four hidden layers of 128 units, ReLU → output layer of 3 with softmax
def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(3,   activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 7. Instantiate and train the ANN (using 10% of training as validation)
input_dim = X_train.shape[1]  # should be 30
model = build_ann_model(input_dim=input_dim)

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.10,    # hold out 10% of training for validation
    epochs=30,                # same number of epochs as paper
    batch_size=64,
    verbose=1
)

# 8. Evaluate on the held‐out test set
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\n=== ANN Test‐Set Classification Report ===")
print(classification_report(y_test, y_pred_classes, target_names=["Distinction","Fail","Pass"]))

print("\n=== ANN Test‐Set Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_classes))

# 9. 10‐Fold Stratified Cross‐Validation (ANN) with 30 features
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), start=1):
    print(f"\nTraining fold {fold}/10...")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    y_tr_cat = to_categorical(y_tr, num_classes=3)
    
    model_cv = build_ann_model(input_dim=input_dim)
    model_cv.fit(
        X_tr, y_tr_cat,
        epochs=30,
        batch_size=64,
        verbose=0
    )
    
    val_pred_probs = model_cv.predict(X_val)
    val_pred_classes = np.argmax(val_pred_probs, axis=1)
    fold_acc = accuracy_score(y_val, val_pred_classes)
    cv_accuracies.append(fold_acc)
    print(f"Fold {fold} accuracy: {fold_acc:.4f}")

print("\n=== ANN 10‐Fold CV Results ===")
print(f"Average CV accuracy: {np.mean(cv_accuracies):.4f}")
print(f"CV accuracy variance: {np.var(cv_accuracies):.6f}")


# In[ ]:


# Step 5: Random Forest Classification with 30 Input Features

# 1. Load the feature‐engineered dataset from Step 3
df = pd.read_csv("Step3_FeatureEngineered_v1.csv")

# 2. Separate X (features) and y (target)
X = df.drop(columns=["final_result"]).values   # 30 input columns
y = df["final_result"].values                  # {0, 1, 2}

# 3. (Optional) Normalize features
#    Random Forests do not require scaling, but we include it for consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into Train/Test (80/20), stratified by y
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Train the Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 6. Evaluate on the held‐out test set
y_pred = rf_model.predict(X_test)

print("\n=== RF Test‐Set Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Distinction","Fail","Pass"]))

print("\n=== RF Test‐Set Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 7. 10‐Fold Stratified Cross‐Validation (RF)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), start=1):
    print(f"\nTraining RF fold {fold}/10...")
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    rf_cv = RandomForestClassifier(
        n_estimators=500,
        criterion='entropy',
        max_depth=None,
        random_state=42
    )
    rf_cv.fit(X_tr, y_tr)
    
    val_pred = rf_cv.predict(X_val)
    fold_acc = accuracy_score(y_val, val_pred)
    rf_accuracies.append(fold_acc)
    print(f"Fold {fold} accuracy: {fold_acc:.4f}")

print("\n=== RF 10‐Fold CV Results ===")
print(f"Average CV accuracy: {np.mean(rf_accuracies):.4f}")
print(f"CV accuracy variance: {np.var(rf_accuracies):.6f}")

