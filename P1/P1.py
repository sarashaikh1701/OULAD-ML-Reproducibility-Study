#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# P1 Reproduction Script
"""
Title: "Ouroboros: early identification of at-risk students without models based on legacy data"
Authors: Martin Hlosta, Zdenek Zdrahal, Jaroslav Zendulka
Source: The Open University, UK

Purpose:
This script reproduces the methodology and results of the above paper using the
Open University Learning Analytics Dataset (OULAD). It executes the complete pipeline described in the paper:
    - Data loading and filtering
    - Feature engineering from current course data only
    - Windowed training and evaluation (day-by-day leading up to A1 cut-off) using:
        * Logistic Regression
        * Support Vector Machine
        * Random Forest
        * Naive Bayes
        * XGBoost
    - PR-AUC calculation as the primary metric
    - Top-K precision/recall analysis
    - XGBoost feature importance extraction
"""

# In[ ]:


import os
import random
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm      import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


# In[ ]:


# RNG seeds
random.seed(42)
np.random.seed(42)


# In[ ]:


student_info       = pd.read_csv("studentInfo.csv")
student_reg        = pd.read_csv("studentRegistration.csv")
student_reg['date_unregistration'].replace(0, np.nan, inplace=True)
student_assessment = pd.read_csv("studentAssessment.csv")
assessments        = pd.read_csv("assessments.csv")
student_vle        = pd.read_csv("studentVle.csv")
vle                = pd.read_csv("vle.csv")
courses            = pd.read_csv("courses.csv")


# In[ ]:


# Step 3: Data Preparation & Filtering

# 3.1 Filter to 2014J and modules B,D,E,F
presentation = '2014J'
modules = ['BBB','DDD','EEE','FFF']
si = student_info[
    (student_info.code_module.isin(modules)) &
    (student_info.code_presentation == presentation)
].copy()

# 3.2 Identify first assessment (A1) per module
asmt = assessments[
    (assessments.code_module.isin(modules)) &
    (assessments.code_presentation == presentation) &
    (assessments.assessment_type == 'TMA') &
    (assessments.weight > 0)
].copy()

first_asmt = (
    asmt.sort_values(['code_module', 'date'])
        .groupby('code_module')
        .first()[['id_assessment', 'date']]
        .rename(columns={'id_assessment': 'A1_id', 'date': 'A1_cutoff'})
        .reset_index()
)

A1_ids = set(first_asmt['A1_id'])

sa = (
    student_assessment
      .merge(assessments[['id_assessment','code_module','code_presentation']],
             on='id_assessment', how='left')
      .query("code_module in @modules and code_presentation == @presentation")
      .query("id_assessment in @A1_ids")
)

# 3.3 Build students frame with cutoff, registration, earliest submission
students = (
    si[['id_student','code_module']]
      .drop_duplicates()
      .merge(first_asmt, on='code_module', how='left')
      .merge(
          student_reg[['id_student','code_module',
                       'date_registration',
                       'date_unregistration']],
          on=['id_student','code_module'], how='left')
)

subs = (
    sa.groupby(['id_student','code_module'])['date_submitted']
      .min()
      .reset_index()
      .rename(columns={'date_submitted':'submission_date'})
)

students = students.merge(subs, on=['id_student','code_module'], how='left')
students['will_submit'] = (
    (students['submission_date'] <= students['A1_cutoff'])
).fillna(False).astype(int)

train_sets = {}
for d in range(12):
    df = students.copy()
    df['prediction_day'] = df['A1_cutoff'] - d
    df['window'] = d
    train_sets[d] = df[['id_student','code_module','prediction_day','A1_cutoff','will_submit','window']]

print(first_asmt)
print(students.head())
print(train_sets[0].head())


# In[ ]:


# Step 4: Feature Engineering

demo_cols = [
    'gender', 'region', 'highest_education', 'imd_band',
    'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
]

def add_vle_stats(df_clicks: pd.DataFrame, prefix: str) -> pd.DataFrame:
    stats = (
        df_clicks
        .groupby(['id_student', 'code_module'])
        .agg(
            first_login   = ('date', 'min'),
            last_login    = ('date', 'max'),
            streak        = ('date', lambda x: (
                               x.sort_values()
                                .diff().eq(1)
                                .groupby((x.diff() != 1).cumsum())
                                .size()
                                .max())),
            total_days    = ('date', 'nunique'),
            sum_clicks    = ('sum_click', 'sum'),
            mean_clicks   = ('sum_click', 'mean'),
            median_clicks = ('sum_click', 'median'),
            max_clicks    = ('sum_click', 'max'),
            min_clicks    = ('sum_click', 'min')
        )
        .reset_index()
    )
    metric_cols = [c for c in stats.columns if c not in ['id_student', 'code_module']]
    return stats.rename(columns={c: f'{prefix}{c}' for c in metric_cols})

def make_pipe(base_clf, X_sample):
    num_cols = X_sample.select_dtypes(include=['number']).columns
    cat_cols = X_sample.select_dtypes(exclude=['number']).columns
    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cast',   FunctionTransformer(lambda x: x.astype(str), validate=False)),
        ('ohe',    OneHotEncoder(handle_unknown='ignore'))
    ])
    prep = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', cat_pipeline,    cat_cols)
    ])
    return Pipeline([('prep', prep), ('clf', clone(base_clf))])


# In[ ]:


# Step 5: Ouroboros Logic

def make_window_core(students_df: pd.DataFrame, d: int) -> pd.DataFrame:
    core = students_df.copy()
    core['prediction_day'] = core['A1_cutoff'] - (d + 1)
    core['window']         = d
    return core

def filter_not_yet_submitted(core: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (core['date_registration'] <= core['prediction_day']) &
        (core['submission_date'].isna() | (core['submission_date'] > core['prediction_day'])) &
        (core['date_unregistration'].isna() | (core['date_unregistration'] >= core['prediction_day']))
    )
    return core.loc[keep].copy()

def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df['will_submit_window'] = (
        (df['submission_date'] > df['prediction_day']) &
        (df['submission_date'] <= df['A1_cutoff'])
    ).fillna(False).astype(int)
    return df

def build_features_for_window(core: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df_dem = core.drop_duplicates(['id_student','code_module']).merge(
        si[['id_student','code_module'] + demo_cols],
        on=['id_student','code_module'], how='left'
    )
    df_reg = df_dem.copy()
    df_reg['days_since_start'] = (df_reg['prediction_day'] - df_reg['date_registration'])
    sv = (
        student_vle
          .merge(vle[['id_site','activity_type']], on='id_site', how='left')
          .merge(df_reg[['id_student','code_module','prediction_day']],
                 on=['id_student','code_module'], how='right')
    )
    sv = sv[sv['date'] <= sv['prediction_day']]
    vle_stats = add_vle_stats(sv[['id_student','code_module','date','sum_click']], prefix='vle_')
    agg = (
        sv.groupby(['id_student','code_module'])
          .agg(total_clicks=('sum_click','sum'),
               active_days =('sum_click', lambda x: (x > 0).sum()),
               avg_clicks  =('sum_click','mean'))
          .reset_index()
    )
    act = (
        sv.groupby(['id_student','code_module','activity_type'])['sum_click']
          .sum()
          .unstack(fill_value=0)
          .reset_index()
    )
    df_vle_agg = (
        df_reg
          .merge(agg,       on=['id_student','code_module'], how='left')
          .merge(act,       on=['id_student','code_module'], how='left')
          .merge(vle_stats, on=['id_student','code_module'], how='left')
          .fillna(0)
    )
    sv_pre = student_vle[
        (student_vle.code_module.isin(modules)) &
        (student_vle.code_presentation == presentation) &
        (student_vle.date < 0)
    ]
    pre_stats = add_vle_stats(sv_pre[['id_student','code_module','date','sum_click']], prefix='pre_')
    pre_sum = (
        sv_pre.groupby(['id_student','code_module'])['sum_click']
              .sum()
              .reset_index()
              .rename(columns={'sum_click':'pre_presentation_clicks'})
    )
    df_pre = (
        df_vle_agg
          .merge(pre_sum,   on=['id_student','code_module'], how='left')
          .merge(pre_stats, on=['id_student','code_module'], how='left')
          .fillna(0)
    )
    N_days = 60
    sv_ta = (
        student_vle
          .merge(vle[['id_site','activity_type']], on='id_site', how='left')
          .merge(core[['id_student','code_module','prediction_day']],
                 on=['id_student','code_module'], how='right')
    )
    sv_ta['rel_day'] = sv_ta['date'] - sv_ta['prediction_day']
    sv_ta = sv_ta[(sv_ta['rel_day'] <= 0) & (sv_ta['rel_day'] >= -N_days)]
    pivot = sv_ta.pivot_table(
        index=['id_student','code_module'],
        columns='rel_day',
        values='sum_click',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    pivot.columns = ['id_student','code_module'] + [f'clicks_day{int(c)}' for c in pivot.columns[2:]]
    df_final = (
        df_pre.merge(pivot, on=['id_student','code_module'], how='left')
              .fillna(0)
              .drop_duplicates(['id_student','code_module'])
    )
    idx = df_final.set_index(['id_student','code_module']).index
    y = (core
         .set_index(['id_student','code_module'])['will_submit_window']
         .groupby(level=[0,1]).first()
         .reindex(idx))
    X = df_final.drop(columns=[
        'id_student','code_module','prediction_day','A1_cutoff',
        'date_registration','date_unregistration',
        'submission_date','will_submit','will_submit_window'
    ], errors='ignore')
    return X.reset_index(drop=True), y.reset_index(drop=True)

core = make_window_core(students, 0)
print("total before filter:", len(core))
print("< today:",  (core['submission_date'] <  core['prediction_day']).sum())
print("= today:",  (core['submission_date'] == core['prediction_day']).sum())
print("> today:",  (core['submission_date'] >  core['prediction_day']).sum(), "|  NaN:",   core['submission_date'].isna().sum())

core = filter_not_yet_submitted(core)
print("after filter:", len(core))
core = add_label(core)
print("positives:", core['will_submit_window'].sum())

# 5.1 Window sanity check
for d in range(12):
    core = make_window_core(students, d)
    core = filter_not_yet_submitted(core)
    core = add_label(core)
    print(f"d={d:2d} rows={len(core):4d}  positives={core['will_submit_window'].sum():4d}")

window_datasets = {}
for d in range(12):
    core = make_window_core(students, d)
    core = filter_not_yet_submitted(core)
    core = add_label(core)
    X_d, y_d = build_features_for_window(core)
    window_datasets[d] = (X_d, y_d)
    print(f"d = {d:2d}   rows = {len(y_d):4d}   positives = {y_d.sum():4d}")


# In[ ]:


# Step 6: Model Training & PR-AUC

models = {
    "LR"  : LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM" : SVC(kernel="rbf", probability=True, class_weight="balanced"),
    "RF"  : RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced"),
    "XGB" : XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, eval_metric="logloss"),
    "NB"  : GaussianNB(),
}

results = []
for d, (X, y) in window_datasets.items():
    if y.sum() == 0 or y.sum() == len(y):
        print(f"⚠︎  window {d} skipped (positives = {y.sum()})")
        continue
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    for name, base_clf in models.items():
        clf = clone(base_clf)
        if name == "XGB":
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            clf.set_params(scale_pos_weight=pos_weight)
        pipe = make_pipe(clf, X_train)
        pipe.fit(X_train, y_train)
        y_score = pipe.predict_proba(X_test)[:, 1]
        pr_auc  = average_precision_score(y_test, y_score)
        results.append({
            "window"   : d,
            "model"    : name,
            "rows"     : len(y_test),
            "positives": int(y_test.sum()),
            "PR_AUC"   : round(pr_auc, 4),
        })
        print(f"d={d:2d}  {name:3s}  PR-AUC = {pr_auc:6.4f}")

summary = (pd.DataFrame(results)
           .pivot(index="window", columns="model", values="PR_AUC")
           .sort_index())
print("\n===  PR-AUC by window ===")
print(summary.to_string(float_format="%.4f"))

student_vle.head()


# In[ ]:


# Step 7: Top-K Precision & Recall

top_ks = [0.05, 0.10, 0.25]
results_topk = []

base_clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    eval_metric="logloss"
)

for d, (X, y) in window_datasets.items():
    if y.sum() in [0, len(y)]:
        continue
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    w = (y_tr==0).sum() / (y_tr==1).sum()
    clf = clone(base_clf)
    clf.set_params(scale_pos_weight=w)
    pipe = make_pipe(clf, X_tr)
    pipe.fit(X_tr, y_tr)
    scores = pipe.predict_proba(X_te)[:,1]
    n = len(y_te)
    for topk in top_ks:
        k = max(1, int(np.floor(topk * n)))
        idx = np.argsort(scores)[-k:]
        y_pred = np.zeros_like(y_te)
        y_pred[idx] = 1
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        results_topk.append({
            "window": d,
            "top_K%": int(topk*100),
            "precision": prec,
            "recall": rec
        })

df_topk = pd.DataFrame(results_topk)
prec_table = df_topk.pivot(index="window", columns="top_K%", values="precision").sort_index()
rec_table  = df_topk.pivot(index="window", columns="top_K%", values="recall").sort_index()
print("=== Top-K Precision ===")
display(prec_table.style.format("{:.3f}"))
print("\n=== Top-K Recall ===")
display(rec_table.style.format("{:.3f}"))


# In[ ]:


# Step 8: Feature Importances (XGBoost)

inspect_windows = [0, 3, 7]
feature_importances = {}

for d in inspect_windows:
    X, y = window_datasets[d]
    w = (y == 0).sum() / (y == 1).sum()
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=w
    )
    pipe = make_pipe(clf, X)
    pipe.fit(X, y)
    ct = pipe.named_steps['prep']
    num_cols = ct.transformers_[0][2]
    cat_pipeline = ct.transformers_[1][1]
    ohe = cat_pipeline.named_steps['ohe']
    cat_cols = ct.transformers_[1][2]
    try:
        cat_names = ohe.get_feature_names(cat_cols)
    except:
        cat_names = ohe.get_feature_names_out(cat_cols)
    feat_names = list(num_cols) + list(cat_names)
    imps = pipe.named_steps['clf'].feature_importances_
    top_idx = imps.argsort()[-5:][::-1]
    feature_importances[d] = [feat_names[i] for i in top_idx]

rows = []
for d, feats in feature_importances.items():
    for rank, feat in enumerate(feats, start=1):
        rows.append({"Window": d, "Rank": rank, "Feature": feat})
df_imp = pd.DataFrame(rows)
pivot = df_imp.pivot(index="Rank", columns="Window", values="Feature")
display(pivot)

