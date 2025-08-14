#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# P2 Reproduction Script
"""
Title: "The Application of Gaussian Mixture Models for the Identification of At-Risk Learners in Massive Open Online Courses"
Authors: Raghad Alshabandar, Abir Jaafar Hussain, Robert Keight, Andy Laws, Thar Baker
Source: Department of Computer Science, Liverpool John Moores University, UK

Purpose:
This script reproduces the methodology and results of the above paper using the 
Open University Learning Analytics Dataset (OULAD), focusing on the BBB-2013B 
presentation. It executes the complete pipeline described in the paper:
    - Data loading and preprocessing
    - Interval-based feature engineering from Virtual Learning Environment (VLE) logs
    - Label creation for on-time vs. late/non-submission
    - Model training and evaluation per interval using:
        * Eigenvalue Decomposition Discriminant Analysis (EDDA)
        * Logistic Regression
        * k-Nearest Neighbors (kNN)
    - Performance aggregation across multiple randomized splits
"""


# In[ ]:


import warnings
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# In[ ]:


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
warnings.filterwarnings("ignore")


# In[ ]:


COURSE = "BBB"
PRES   = "2013B"
DATA_DIR = Path(".")
CSV_OUT = Path("all_metrics.csv")
RIDGE      = 1e-3
N_REPEATS  = 100


# In[ ]:


# Step 2: Data Loading and Manipulation 
read = lambda f: pd.read_csv(DATA_DIR / f)

student_info       = read("studentInfo.csv")
student_vle        = read("studentVle.csv")
assessments        = read("assessments.csv")
student_assessment = read("studentAssessment.csv")
vle                = read("vle.csv")

student_assessment = student_assessment.merge(
    assessments[["id_assessment","code_module","code_presentation"]],
    on="id_assessment", how="left"
)

student_vle = student_vle.merge(vle[["id_site","activity_type"]], on="id_site", how="left")

mask = (student_info.code_module == COURSE) & (student_info.code_presentation == PRES)
filtered_info   = student_info.loc[mask]

mask = (student_vle.code_module == COURSE) & (student_vle.code_presentation == PRES)
filtered_vle    = student_vle.loc[mask]

mask = (assessments.code_module == COURSE) & (assessments.code_presentation == PRES)
filtered_assessments = assessments.loc[mask]

mask = (student_assessment.code_module == COURSE) & (student_assessment.code_presentation == PRES)
filtered_student_assess = student_assessment.loc[mask]

print(f"✓ Loaded course {COURSE}-{PRES}")
print("  studentInfo       :", filtered_info.shape)
print("  studentVle        :", filtered_vle.shape)
print("  assessments       :", filtered_assessments.shape)
print("  studentAssessment :", filtered_student_assess.shape)

bbb_students = (
    filtered_info[["id_student","code_module","code_presentation"]]
      .drop_duplicates()
)


# In[ ]:


#Step 3 - defining time intervals
deadline_dates = (
    filtered_assessments
      .query("assessment_type in ['TMA','CMA'] and weight > 0")
      .sort_values("date")
      .drop_duplicates(subset="date")
      ["date"]
      .head(6)
      .tolist()
)
if len(deadline_dates) < 6:
    raise ValueError("Fewer than six graded deadlines found for this course!")
print("✓ Deadlines (days since course start):", deadline_dates)

assignments_by_interval = (
    filtered_assessments
      .query("date in @deadline_dates")
      .sort_values("date")
      .groupby("date")
      .head(1)
      .reset_index(drop=True)
)
assignments_by_interval["interval"] = np.arange(1, 7)

edges = [-np.inf] + deadline_dates
def day_to_interval(day):
    for i in range(1, len(edges)):
        if day <= edges[i]:
            return i
    return 6

filtered_vle["interval"] = filtered_vle["date"].apply(day_to_interval)

print("\nClicks per interval")
print(filtered_vle["interval"].value_counts().sort_index())
print("\nassignments_by_interval:")
display(assignments_by_interval[["interval","id_assessment","date"]])


# In[ ]:


#Step 4: Feature Engineering
canonical = [
    "homepage","resource","url","forumng","oucontent","subpage",
    "quiz","oucollaborate","externaltool","ouwiki","dataplus",
    "ouelluminate","oublog","checkmark","scheduler","sharedsubpage",
    "page","repeatactivity","htmlactivity","feedback","glossary",
    "dualpane"
]

filtered_vle = filtered_vle.copy()
filtered_vle["interval"] = filtered_vle["date"].apply(day_to_interval)

session_counts = (
    filtered_vle
      .groupby(["id_student", "interval", "activity_type"])
      .size()
      .reset_index(name="num_sessions")
)
click_counts = (
    filtered_vle
      .groupby(["id_student", "interval", "activity_type"])["sum_click"]
      .sum()
      .reset_index(name="total_clicks")
)

feat_long = session_counts.merge(click_counts, on=["id_student", "interval", "activity_type"])
sessions_w = (
    feat_long.pivot_table(index=["id_student", "interval"],
                          columns="activity_type",
                          values="num_sessions",
                          fill_value=0)
)
clicks_w = (
    feat_long.pivot_table(index=["id_student", "interval"],
                          columns="activity_type",
                          values="total_clicks",
                          fill_value=0)
)

sessions_w.columns = [f"session_{c}" for c in sessions_w.columns]
clicks_w.columns   = [f"clicks_{c}"  for c in clicks_w.columns]

full_idx = pd.MultiIndex.from_product(
    [bbb_students["id_student"].unique(), range(1, 7)],
    names=["id_student", "interval"]
).to_frame(index=False)

features = (
    full_idx
      .merge(sessions_w.reset_index(), how="left")
      .merge(clicks_w.reset_index(),  how="left")
      .fillna(0)
)

all_types = vle["activity_type"].unique()
for t in all_types:
    for prefix in ("session_", "clicks_"):
        col = f"{prefix}{t}"
        if col not in features.columns:
            features[col] = 0
for t in canonical:
    for prefix in ("session_", "clicks_"):
        col = f"{prefix}{t}"
        if col not in features.columns:
            features[col] = 0

keep_cols = (
    ["id_student", "interval"] +
    [f"session_{t}" for t in canonical] +
    [f"clicks_{t}"  for t in canonical]
)
final_features = features[keep_cols].copy()
print(" final_features shape:", final_features.shape)
print("  (should be 6 × #students rows, 46 columns)")


# In[ ]:


#Step 5: Label generation

tma_pool = (
    filtered_assessments
      .query("assessment_type == 'TMA' and weight > 0")
      .sort_values("date")
      .head(6)
      .reset_index(drop=True)
)
tma_pool["interval"] = range(1, 7)
print("✓ Deadlines (days since course start):", tma_pool["date"].tolist())

sa_clean = (
    filtered_student_assess
      .loc[filtered_student_assess["is_banked"] == 0]
      .dropna(subset=["date_submitted"])
)
sa_merged = (
    sa_clean
      .merge(tma_pool[["id_assessment", "interval", "date"]],
             on="id_assessment", how="inner", validate="many_to_one")
)
first_sub = (
    sa_merged
      .groupby(["id_student", "interval"], as_index=False)["date_submitted"]
      .min()
)
first_sub = first_sub.merge(tma_pool[["interval", "date"]], on="interval")
first_sub["on_time"] = first_sub["date_submitted"] <= first_sub["date"]

label_grid = (
    pd.MultiIndex.from_product(
        [bbb_students["id_student"].unique(), range(1, 7)],
        names=["id_student", "interval"]
    ).to_frame(index=False)
      .merge(first_sub[["id_student", "interval", "on_time"]],
             on=["id_student", "interval"], how="left")
      .fillna({"on_time": False})
)
label_grid["Y"] = 1 - label_grid["on_time"].astype(int)
labels = label_grid[["id_student", "interval", "Y"]]

final_features = final_features.merge(labels, on=["id_student", "interval"], how="left")
print("\nDistribution of Y per interval (fractions):")
print((final_features.groupby("interval")["Y"]
       .value_counts(normalize=True)
       .unstack()
       .round(3)))


# In[ ]:


#Step 6: Functions
def make_slice(full_df, t):
    cur = full_df[full_df.interval == t].set_index("id_student")
    if t == 1:
        return cur.reset_index()
    prev = full_df[full_df.interval == t-1].set_index("id_student").add_suffix("_prev")
    return cur.join(prev, how="left").fillna(0).reset_index()

def log_lik(clf, X, y):
    lp = clf.predict_log_proba(X)
    return lp[np.arange(len(y)), y].sum()

def bic(loglik, free_params, n):
    return -2*loglik + free_params*np.log(n)


# In[1]:


#Step 7: Modelling
all_rows = []
for t in range(1, 7):
    df_t   = make_slice(final_features, t)
    X_full = df_t.filter(regex="^(session_|clicks_)").values
    y_full = df_t["Y"].values

    active = X_full.sum(axis=1) > 0
    X_full, y_full = X_full[active], y_full[active]

    if len(np.unique(y_full)) < 2:
        print(f"⚠︎  t={t}: only one class present → skipped")
        continue

    for run in range(N_REPEATS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, test_size=.40, stratify=y_full,
            random_state=GLOBAL_SEED + run
        )

        eee = Pipeline([
            ("sc",  StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=RIDGE))
        ]).fit(X_tr, y_tr)

        vvv = Pipeline([
            ("sc",  StandardScaler()),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=RIDGE, store_covariance=True))
        ]).fit(X_tr, y_tr)

        n, p, k = X_tr.shape[0], X_tr.shape[1], 2
        bic_eee = bic(log_lik(eee, X_tr, y_tr), p*(p+1)/2 + k*p, n)
        bic_vvv = bic(log_lik(vvv, X_tr, y_tr), k*(p + p*(p+1)/2), n)
        edda_best = eee if bic_eee < bic_vvv else vvv

        logreg = Pipeline([
            ("sc",  StandardScaler()),
            ("lr",  LogisticRegression(C=10, max_iter=500, class_weight="balanced"))
        ]).fit(X_tr, y_tr)

        knn = Pipeline([
            ("sc",  StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=10))
        ]).fit(X_tr, y_tr)

        for name, mdl in {"EDDA": edda_best, "LogReg": logreg, "kNN": knn}.items():
            prob = mdl.predict_proba(X_te)[:, 1]
            pred = mdl.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            sens = tp / (tp + fn) if tp + fn else 0
            spec = tn / (tn + fp) if tn + fp else 0

            all_rows.append({
                "interval":    t,
                "model":       name,
                "accuracy":    accuracy_score(y_te, pred),
                "f1":          f1_score(y_te, pred, zero_division=0),
                "sensitivity": sens,
                "specificity": spec,
                "auc":         roc_auc_score(y_te, prob)
            })

    print(f"finished t={t}")


# In[ ]:


#Step 8: metric saves
metrics_df = (pd.DataFrame(all_rows)
                .groupby(["interval", "model"])
                .median()
                .reset_index()
                .sort_values(["interval", "model"]))
metrics_df.to_csv(CSV_OUT, index=False)
display(metrics_df.round(3))
print(f"metrics (median over {N_REPEATS} splits) → {CSV_OUT}")


# In[ ]:




