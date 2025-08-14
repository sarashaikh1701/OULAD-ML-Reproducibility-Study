#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Imports and data loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans

info = pd.read_csv("studentInfo.csv")
asmt = pd.read_csv("studentAssessment.csv")
vle = pd.read_csv("studentVle.csv")
asmt.head()
info.head()
assess = pd.read_csv("assessments.csv")
assess.head()


# In[ ]:


# Step 2: Data preprocessing
filt = info[info["final_result"] != "Withdrawn"].copy()
banked = (asmt[asmt["is_banked"] != 0]
            .merge(assess[["id_assessment", "code_module", "code_presentation"]],
                   on="id_assessment", how="left")
            [["code_module", "code_presentation", "id_student"]]
            .drop_duplicates())
filt = (filt.merge(banked.assign(_drop=True),
                   on=["code_module", "code_presentation", "id_student"],
                   how="left")
            .query("_drop.isna()")
            .drop(columns=["_drop"]))
filt = filt.dropna(subset=["code_module", "code_presentation", "id_student", "final_result"])
LABEL_MAP = {"Pass": 1, "Distinction": 1, "Fail": 0}
filt["label"] = filt["final_result"].map(LABEL_MAP).astype(int)
df_filtered = filt[["code_module", "code_presentation", "id_student", "label"]]
df_filtered.head()


# In[ ]:


# Step 3: Activity vectors
TIME_WINDOW = 245
DAY_MIN, DAY_MAX = 0, 244
vle = vle[(vle["date"] >= DAY_MIN) & (vle["date"] <= DAY_MAX)].copy()
pivot = vle.pivot_table(index=["code_module", "code_presentation", "id_student"],
                        columns="date",
                        values="sum_click",
                        aggfunc="sum",
                        fill_value=0)
for d in range(TIME_WINDOW):
    if d not in pivot.columns:
        pivot[d] = 0
pivot = pivot.sort_index(axis=1)
clicks_matrix = pivot.values.astype(np.int32)
inter_matrix  = (clicks_matrix > 0).astype(np.int8)
max_clicks = clicks_matrix.max(axis=1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    norm_matrix = np.where(max_clicks == 0, 0, clicks_matrix / max_clicks)
df_vectors = pivot.reset_index()[["code_module", "code_presentation", "id_student"]].copy()
df_vectors["clicks_vector"]       = list(clicks_matrix)
df_vectors["interactions_vector"] = list(inter_matrix)
df_vectors["normalized_vector"]   = list(norm_matrix)
dataset = df_filtered.merge(df_vectors,
                            on=["code_module", "code_presentation", "id_student"],
                            how="inner")
print(f"Shape: {dataset.shape}")
dataset.head()
missing_any = dataset.isnull().any(axis=1).sum()
missing_any


# In[ ]:


# Step 4: Feature sets
KEYS = ["code_module", "code_presentation", "id_student"]
y      = dataset["label"].values
key_df = dataset[KEYS]
print("Label vector shape:", y.shape, "| positive class share:", y.mean().round(4))

cols_cat = ["age_band", "disability", "gender", "highest_education", "region"]
cols_num = ["num_of_prev_attempts", "studied_credits"]
stu_raw  = info[KEYS + cols_cat + cols_num]
X_info = (key_df
          .merge(stu_raw, on=KEYS, how="left")
          .pipe(lambda df: pd.concat([
                 pd.get_dummies(df[cols_cat]),
                 df[cols_num].fillna(0)
          ], axis=1))
          .fillna(0)
          .values.astype(float))
print("Student-info feature matrix:", X_info.shape)

X_count = np.vstack(dataset["clicks_vector"].values).astype(float)
X_bin   = np.vstack(dataset["interactions_vector"].values).astype(float)
X_norm  = np.vstack(dataset["normalized_vector"].values).astype(float)
print("Activity matrices  |  count:", X_count.shape,
      "| binary:", X_bin.shape,
      "| normalized:", X_norm.shape)

feature_sets = {
    "activity_count"  : X_count,
    "info+binary"     : hstack([X_info, X_bin]),
    "info+normal"     : hstack([X_info, X_norm]),
    "info+count"      : hstack([X_info, X_count])
}
for k, v in feature_sets.items():
    print(f"{k:<15} → {v.shape}")


# In[ ]:


# Step 5: Models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorers = {"acc":  make_scorer(accuracy_score),
           "prec": make_scorer(precision_score),
           "rec":  make_scorer(recall_score),
           "f1":   make_scorer(f1_score)}

clf = DecisionTreeClassifier(random_state=0)
for name, X in feature_sets.items():
    scores = cross_validate(clf, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    print(f"{name:<15}  "
          f"Acc {scores['test_acc'].mean():.3f}  "
          f"Prec {scores['test_prec'].mean():.3f}  "
          f"Rec {scores['test_rec'].mean():.3f}  "
          f"F1 {scores['test_f1'].mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorers = {"acc":  make_scorer(accuracy_score),
           "prec": make_scorer(precision_score),
           "rec":  make_scorer(recall_score),
           "f1":   make_scorer(f1_score)}
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
for name, X in feature_sets.items():
    scores = cross_validate(clf, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    print(f"{name:<15}  "
          f"Acc {scores['test_acc'].mean():.3f}  "
          f"Prec {scores['test_prec'].mean():.3f}  "
          f"Rec {scores['test_rec'].mean():.3f}  "
          f"F1 {scores['test_f1'].mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorers = {"acc":  make_scorer(accuracy_score),
           "prec": make_scorer(precision_score),
           "rec":  make_scorer(recall_score),
           "f1":   make_scorer(f1_score)}
clf = LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000, random_state=0)
for name, X in feature_sets.items():
    scores = cross_validate(clf, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    print(f"{name:<15}  "
          f"Acc {scores['test_acc'].mean():.3f}  "
          f"Prec {scores['test_prec'].mean():.3f}  "
          f"Rec {scores['test_rec'].mean():.3f}  "
          f"F1 {scores['test_f1'].mean():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scorers = {"acc":  make_scorer(accuracy_score),
           "prec": make_scorer(precision_score),
           "rec":  make_scorer(recall_score),
           "f1":   make_scorer(f1_score)}
clf = SVC(kernel="rbf", gamma="auto", C=1.0, random_state=0)
for name, X in feature_sets.items():
    scores = cross_validate(clf, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    print(f"{name:<15}  "
          f"Acc {scores['test_acc'].mean():.3f}  "
          f"Prec {scores['test_prec'].mean():.3f}  "
          f"Rec {scores['test_rec'].mean():.3f}  "
          f"F1 {scores['test_f1'].mean():.3f}")


# In[ ]:


# Step 6: Clustering
days = np.arange(X_bin.shape[1])
kmeans = KMeans(n_clusters=9, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(X_bin)
inter_df = pd.DataFrame(X_bin, columns=days)
inter_df["cluster"] = cluster_labels
cluster_curves = inter_df.groupby("cluster").mean().sort_index()
print("Cluster sizes:", inter_df["cluster"].value_counts().sort_index().tolist())
cluster_curves.head()

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    ax.plot(days, cluster_curves.loc[i].values)
    ax.set_title(f"Cluster {i}  (n={inter_df.cluster.value_counts()[i]})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
fig.suptitle("Mean binary activity curves per K-Means cluster")
plt.tight_layout()
plt.show()

order = cluster_curves.mean(axis=1).sort_values(ascending=False).index
plt.figure(figsize=(12, 7))
for i in order:
    curve = cluster_curves.loc[i].rolling(window=7, center=True, min_periods=1).mean()
    plt.plot(days, curve, label=f"Cluster {i}")
plt.xlabel("Course day (0–244)")
plt.ylabel("7-day smoothed mean interaction")
plt.title("K-Means clusters (ordered, smoothed)")
plt.legend(ncol=3)
plt.tight_layout()
plt.show()

