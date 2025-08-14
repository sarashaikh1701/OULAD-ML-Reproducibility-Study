#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Paper 6 Reproduction Script
"""
Title: "Predicting academic performance of students from VLE big data using deep learning models"
Authors: Md Shoaib Ahmed, Shazia Sadiq, Ashad Kabir, Shazia W. Sadiq, Rolf A. Schwitter (2021)

Purpose:
End-to-end entry-point script for academic performance prediction:
    - Data loading and preprocessing
    - Feature engineering
    - Train/test split
    - Deep learning model training and evaluation
    - Save trained model weights
"""


# In[ ]:


# Step 1: Imports + RNG seeds
import os, json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, roc_auc_score,
                             accuracy_score, precision_score, recall_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[ ]:


# Step 2: Load data
si   = pd.read_csv('studentInfo.csv')
svle = pd.read_csv('studentVle.csv')
vle  = pd.read_csv('vle.csv')
sa   = pd.read_csv('studentAssessment.csv')
ass  = pd.read_csv('assessments.csv')
crs  = pd.read_csv('courses.csv')


# In[ ]:


# Step 3: Feature engineering (clicks, assessments, master table)
svle = svle.merge(vle[['id_site','activity_type']], on='id_site', how='left')

clicks = (
    svle.groupby(['id_student','code_module','code_presentation'])['sum_click']
        .sum()
        .reset_index(name='total_clicks')
)

sa_aug = sa.merge(ass, on='id_assessment', how='left')
ass_agg = (
    sa_aug.groupby(['id_student','code_module','code_presentation'])
          .agg(avg_score=('score','mean'),
               n_submissions=('id_assessment','size'))
          .reset_index()
)

data = (
    si.merge(clicks,  on=['id_student','code_module','code_presentation'], how='left')
      .merge(ass_agg, on=['id_student','code_module','code_presentation'], how='left')
)
data = data.dropna(subset=['final_result'])

def task_pass_fail(df):
    subset = df[df.final_result.isin(['Pass','Distinction','Fail'])].copy()
    subset['label'] = (subset.final_result == 'Fail').astype(int)
    return subset

def task_withdrawn_pass(df):
    subset = df[df.final_result.isin(['Pass','Distinction','Withdrawn'])].copy()
    subset['label'] = (subset.final_result == 'Withdrawn').astype(int)
    return subset

def task_dist_pass(df):
    subset = df[df.final_result.isin(['Pass','Distinction'])].copy()
    subset['label'] = (subset.final_result == 'Distinction').astype(int)
    return subset

def task_dist_fail(df):
    subset = df[df.final_result.isin(['Distinction','Fail'])].copy()
    subset['label'] = (subset.final_result == 'Distinction').astype(int)
    return subset

tasks = {
    'Pass_vs_Fail'        : task_pass_fail(data),
    'Withdrawn_vs_Pass'   : task_withdrawn_pass(data),
    'Distinction_vs_Pass' : task_dist_pass(data),
    'Distinction_vs_Fail' : task_dist_fail(data)
}


# In[ ]:


# Step 4: Pipeline matrices
imputer = SimpleImputer(strategy='constant', fill_value=0)
scaler  = MinMaxScaler()
task_matrices = {}

for name, df in tasks.items():
    y = df['label'].values
    num_cols = df.select_dtypes(include='number').columns.difference(['id_student','label'])
    X_num = imputer.fit_transform(df[num_cols])
    X_num = scaler.fit_transform(X_num)
    cat_cols = df.select_dtypes(include='object').columns.difference(['final_result'])
    X_cat = pd.get_dummies(df[cat_cols], dummy_na=True, drop_first=True)
    X_full = np.hstack([X_num, X_cat.values])
    if X_full.shape[1] > 30:
        svd = TruncatedSVD(n_components=30, random_state=SEED)
        X_red = svd.fit_transform(X_full)
        X_df = pd.DataFrame(X_red, index=df.index, columns=[f'F{i+1}' for i in range(30)])
    else:
        X_df = pd.DataFrame(X_full, index=df.index,
                            columns=[f'X{i+1}' for i in range(X_full.shape[1])])
    task_matrices[name] = X_df.assign(label=y)

for k, m in task_matrices.items():
    print(f'{k:22s} -> {m.shape}')

# Step 4.1: Static features (54 -> SVD 30)
static = data[['id_student','code_module','code_presentation',
               'highest_education','imd_band','age_band',
               'num_of_prev_attempts','studied_credits',
               'disability','gender','region']]

svle_split = svle.copy()

def click_totals(df, mask, prefix):
    sub = df[mask]
    tot  = (sub.groupby(['id_student','code_module','code_presentation'])['sum_click']
              .sum().rename(f'{prefix}T_Clicks'))
    acts = (sub.pivot_table(index=['id_student','code_module','code_presentation'],
                            columns='activity_type', values='sum_click', aggfunc='sum')
              .add_prefix(prefix))
    return pd.concat([tot, acts], axis=1).reset_index()

before = click_totals(svle_split, svle_split['date'] < 0,  'BC_')
after  = click_totals(svle_split, svle_split['date'] >= 0, 'AC_')

ass_flags = (
    sa_aug.assign(late = sa_aug['date_submitted'] > sa_aug['date'])
          .groupby(['id_student','code_module','code_presentation'])
          .agg(ModuleAsigns=('id_assessment','nunique'),
               LateAsignsSub=('late','sum'))
          .reset_index()
)

sa_dead = sa_aug[['id_student','code_module','code_presentation',
                  'id_assessment','date']].rename(columns={'date':'deadline_day'})

click_base = svle[['id_student','code_module','code_presentation',
                   'date','sum_click']].rename(columns={'date':'click_day'})

def clicks_window(offset, col_name):
    merged = sa_dead.merge(click_base, on=['id_student','code_module','code_presentation'], how='left')
    mask   = merged['click_day'] == merged['deadline_day'] + offset
    out    = (merged.loc[mask]
                    .groupby(['id_student','code_module','code_presentation'])['sum_click']
                    .sum()
                    .rename(col_name)
                    .reset_index())
    return out

F25 = clicks_window(-1,  'PreA_1')
F27 = clicks_window( 0,  'OnAsClicks')
F30 = clicks_window(10,  'PostA_10')

keys = ['id_student','code_module','code_presentation']
static_full = (static
               .merge(before, on=keys, how='left')
               .merge(after,  on=keys, how='left')
               .merge(ass_flags, on=keys, how='left')
               .merge(clicks.rename(columns={'total_clicks':'Tc_Activity'}), on=keys, how='left')
               .merge(F25, on=keys, how='left')
               .merge(F27, on=keys, how='left')
               .merge(F30, on=keys, how='left')
               .fillna(0))

X = static_full.drop(columns=keys)
X = pd.get_dummies(X, dummy_na=True, drop_first=True)
X = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X)
X = MinMaxScaler().fit_transform(X)
static_SVD30 = TruncatedSVD(n_components=30, random_state=SEED).fit_transform(X)
print("Static feature matrix:", static_SVD30.shape)

# Step 4.2: Temporal features (quartiles × 20 acts)
master_idx = data[keys].drop_duplicates().set_index(keys)
activity_types = sorted(vle['activity_type'].dropna().unique().tolist())

len_cols = [c for c in crs.columns if 'length' in c.lower()]
if len_cols:
    len_col = len_cols[0]
    pres_len = crs[['code_module','code_presentation', len_col]].rename(columns={len_col: 'length'})
else:
    pres_len = (svle.groupby(['code_module','code_presentation'])['date']
                      .max().reset_index(name='length'))

svle_len = svle.merge(pres_len, on=['code_module','code_presentation'], how='left', validate='m:1')

def make_quartile(q_num):
    frac  = q_num / 4.0
    label = f'Q{q_num}'
    sub = svle_len[svle_len['date'] <= svle_len['length']*frac]
    clicks_q = (sub.groupby(keys + ['activity_type'])['sum_click']
                   .sum().unstack(fill_value=0)
                   .reindex(columns=activity_types, fill_value=0)
                   .add_prefix(f'{label}_'))
    full = master_idx.join(clicks_q, how='left').fillna(0).reset_index()
    return full

temporal_frames = [make_quartile(q) for q in range(1, 5)]
for lbl, df_ in zip(['Q1','Q2','Q3','Q4'], temporal_frames):
    print(f'{lbl} shape:', df_.shape)

assert static_full.drop(columns=keys).shape[1] == 54
for q in temporal_frames:
    assert q.filter(regex='^Q[1-4]_').shape[1] == 20


# In[ ]:


# Step 5: Deep ANN for four tasks
def activity_only(df, prefix):
    df_idx = df.set_index(keys)
    acts = df_idx.filter(regex=f'^{prefix}')
    return acts.astype('float32')

X_static = (pd.DataFrame(static_SVD30,
                         columns=[f'S{i+1}' for i in range(30)],
                         index=static_full.set_index(keys).index)
            .astype('float32'))
X_q4 = activity_only(temporal_frames[3], 'Q4_')
X_q2 = activity_only(temporal_frames[1], 'Q2_')
X_hybrid = X_static.join(X_q2, how='left').fillna(0.0)

feature_sets = {'STATIC-30': X_static, 'TEMP-Q4': X_q4, 'HYBRID-50': X_hybrid}

def safe_name(txt):
    return txt.replace(" ", "").replace("|", "__")

def paths(task, fset):
    base = safe_name(f"{task}__{fset}")
    return f"logs/{base}.csv", f"weights/{base}.h5"

def build_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])

def resume_or_fit(model, Xtr, ytr, Xval, yval,
                  csv_path, w_path,
                  epochs=100, batch_size=64, class_wt=None):
    initial_epoch = 0
    if os.path.exists(csv_path):
        initial_epoch = pd.read_csv(csv_path)['epoch'].max() + 1
        if initial_epoch >= epochs:
            print("✔  Training already complete – skipping")
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(w_path),  exist_ok=True)

    csv_cb = tf.keras.callbacks.CSVLogger(csv_path, append=os.path.exists(csv_path))
    es_cb  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])

    if os.path.exists(w_path):
        model.load_weights(w_path)

    if initial_epoch < epochs:
        model.fit(Xtr, ytr,
                  validation_data=(Xval, yval),
                  epochs=epochs,
                  batch_size=batch_size,
                  class_weight=class_wt,
                  callbacks=[csv_cb, es_cb],
                  initial_epoch=initial_epoch,
                  verbose=2)
        model.save_weights(w_path)

    return model

results = defaultdict(dict)

for task_name, task_df in tasks.items():
    y = task_df['label'].astype('float32').values
    mi = pd.MultiIndex.from_frame(task_df[keys])

    for fset_name, Xmat in feature_sets.items():
        X = Xmat.loc[mi].values.astype('float32')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
        cw = compute_class_weight('balanced', classes=[0, 1], y=y_tr)
        class_wt = {0: cw[0], 1: cw[1]}

        ann = build_model(X.shape[1])
        csv_path, w_path = paths(task_name, fset_name)
        ann = resume_or_fit(ann, X_tr, y_tr, X_te, y_te,
                            csv_path, w_path, epochs=100, batch_size=64, class_wt=class_wt)

        eval_vals = ann.evaluate(X_te, y_te, verbose=0)
        raw = dict(zip(ann.metrics_names, eval_vals))
        scores = {}
        for k, v in raw.items():
            base = k.split('_')[0]
            scores[base] = float(v)
        for k in ['accuracy', 'precision', 'recall', 'auc']:
            scores.setdefault(k, np.nan)

        results[task_name][fset_name] = scores

        print(f"{task_name:<18} | {fset_name:<9} | "
              f"Acc {scores['accuracy']:.4f}  "
              f"Prec {scores['precision']:.4f}  "
              f"Rec {scores['recall']:.4f}  "
              f"AUC {scores['auc']:.4f}")


# In[ ]:


# Step 6: Baseline models
baselines = {
    'LogisticRegression': LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED),
    'SVM-RBF'           : SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
}

print("\n=== Baseline results ===")
for task_name, task_df in tasks.items():
    y = task_df['label'].values
    idx = pd.MultiIndex.from_frame(task_df[keys])
    for fset_name, Xmat in feature_sets.items():
        X = Xmat.loc[idx].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
        for model_name, model in baselines.items():
            model.fit(X_tr, y_tr)
            y_pred  = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:,1]
            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec  = recall_score(y_te, y_pred, zero_division=0)
            auc  = roc_auc_score(y_te, y_proba)
            print(f"{task_name:18s} | {fset_name:9s} | {model_name:18s} | "
                  f"Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  AUC {auc:.4f}")

# Step 6.1: Baseline repeated splits (static features)
def eval_baseline(X, y, classifier, n_splits=10, test_size=0.3, random_state=SEED):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    accs, precs, recs, losses = [], [], [], []
    for train_idx, test_idx in sss.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        clf = classifier.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        preds = clf.predict(Xte)
        accs.append(accuracy_score(yte, preds))
        precs.append(precision_score(yte, preds))
        recs.append(recall_score(yte, preds))
        losses.append(log_loss(yte, proba))
    return {'accuracy': np.mean(accs)*100, 'loss': np.mean(losses),
            'precision': np.mean(precs), 'recall': np.mean(recs)}

lr  = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=SEED)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED)

results_tbl = []
for task_name, df_ in tasks.items():
    mi = pd.MultiIndex.from_frame(df_[['id_student','code_module','code_presentation']])
    Xs = X_static.loc[mi].values
    yy = df_['label'].values
    for name, clf in [('LR', lr), ('SVM-RBF', svm)]:
        scores = eval_baseline(Xs, yy, clf, n_splits=10)
        results_tbl.append({
            'Category': task_name.replace('_','/'),
            'Technique': name,
            'Accuracy (%)': scores['accuracy'],
            'Loss': scores['loss'],
            'Precision': scores['precision'],
            'Recall': scores['recall']
        })

baseline_df = pd.DataFrame(results_tbl)
print(baseline_df.pivot(index='Category', columns='Technique'))


# In[ ]:


# Step 7: Early prediction (quarterly)
import matplotlib.pyplot as plt

quartile_names = ['Q1','Q1-2','Q1-3','Q1-4']
cum_frames = temporal_frames

def get_activity_matrix(df, task_df):
    idx = pd.MultiIndex.from_frame(task_df[['id_student','code_module','code_presentation']])
    Xq  = df.set_index(['id_student','code_module','code_presentation'])
    return Xq.loc[idx].astype('float32').values

ep_results = {t:{'acc':[], 'loss':[]} for t in tasks}

for task_name, task_df in tasks.items():
    y = task_df['label'].astype('float32').values
    for q_df in cum_frames:
        X = get_activity_matrix(q_df, task_df)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
        cw = compute_class_weight('balanced', classes=[0,1], y=ytr)
        class_wt = {0: cw[0], 1: cw[1]}
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(Xtr.shape[1],)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1,  activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(Xtr, ytr, validation_split=0.2, epochs=100, batch_size=64,
                  class_weight=class_wt, callbacks=[es], verbose=0)
        loss, acc = model.evaluate(Xte, yte, verbose=0)
        ep_results[task_name]['acc'].append(acc)
        ep_results[task_name]['loss'].append(loss)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for t, metrics in ep_results.items():
    plt.plot(quartile_names, metrics['acc'], marker='o', label=t)
plt.title('Quarterly Test Accuracy')
plt.xlabel('Cumulative Quartile')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=15)

plt.subplot(1,2,2)
for t, metrics in ep_results.items():
    plt.plot(quartile_names, metrics['loss'], marker='o', label=t)
plt.title('Quarterly Test Loss')
plt.xlabel('Cumulative Quartile')
plt.ylabel('Loss')
plt.legend()
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

