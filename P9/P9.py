#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# P9 Reproduction Script
"""
Title: "How Well a Student Performed: A Machine Learning Approach to Classify Students’ Performance on Virtual Learning Environment"
Authors: Faisal M. Alnassar, Eesa Alshraideh, Eman Alnassar (2021)

Purpose:
End-to-end entry-point script for binary student performance classification using OULAD:
    - Step 2: Data loading, preprocessing, and merging demographic, engagement, and performance features
    - Step 3: Feature-set definitions (D, E, P and combinations)
    - Step 4: Model training and evaluation with k-NN, SVC, and ANN
    - Metrics: Accuracy, F1-score, and J-Index (Jaccard)
"""


# In[ ]:


#Step 1

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from IPython.display import display

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)


# In[ ]:


#Step 2: Data preprocessing
DATA_DIR = Path('.')

info = pd.read_csv(DATA_DIR / 'studentInfo.csv')
student_vle = pd.read_csv(DATA_DIR / 'studentVle.csv')
assessments = pd.read_csv(DATA_DIR / 'assessments.csv')
student_assess = pd.read_csv(DATA_DIR / 'studentAssessment.csv')

info = (info
        .query("final_result in ['Pass','Distinction','Fail','Withdrawn']")
        .assign(label=lambda d: d['final_result'].isin(['Pass','Distinction']).astype(int))
        .drop(columns=['final_result',
                       'date_registration','date_unregistration',
                       'studied_credits'], errors='ignore'))

ROW_KEY = ['id_student', 'code_module', 'code_presentation']

eng = (student_vle
       .groupby(ROW_KEY)['sum_click']
       .agg(['sum', 'mean'])
       .reset_index()
       .rename(columns={'sum': 'sum_clicks',
                        'mean': 'avg_clicks_per_day'}))

sa = (student_assess
      .merge(assessments[['id_assessment','assessment_type',
                          'code_module','code_presentation']],
             on='id_assessment', how='left'))

perf = (sa.groupby(ROW_KEY)['score']
          .agg(['mean', 'count', lambda x: (x >= 40).sum()])
          .reset_index())
perf.columns = ROW_KEY + ['avg_score','num_assessments','num_passed_assessments']

exam_scores = (sa[sa['assessment_type'] == 'Exam']
                 .groupby(ROW_KEY)['score']
                 .max()
                 .reset_index()
                 .rename(columns={'score':'exam_score'}))

perf = perf.merge(exam_scores, on=ROW_KEY, how='left')

df = (info
      .merge(eng,  on=ROW_KEY, how='left')
      .merge(perf, on=ROW_KEY, how='left')
      .fillna(0)
      .reset_index(drop=True))

print('Step 2 complete - shape:', df.shape)
display(df.head())
df.to_csv('oulad_step2_preprocessed.csv', index=False)


# In[ ]:


#Step 3: Feature sets and Model pipelines
df = pd.read_csv('oulad_step2_preprocessed.csv')

demographic_cols = ['gender','age_band','highest_education','disability','region']
engagement_cols  = ['sum_clicks','avg_clicks_per_day']
performance_cols = ['avg_score','num_assessments','num_passed_assessments','exam_score']

feature_sets = {
    'D'     : demographic_cols,
    'E'     : engagement_cols,
    'P'     : performance_cols,
    'D+E'   : demographic_cols + engagement_cols,
    'D+P'   : demographic_cols + performance_cols,
    'E+P'   : engagement_cols  + performance_cols,
    'D+E+P' : demographic_cols + engagement_cols + performance_cols
}

def make_pipeline(cat_cols, num_cols, estimator):
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value=0)),
            ('sc',  StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh',  OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    n_estimated = len(num_cols)
    if cat_cols:
        n_estimated += 2 * len(cat_cols)
    k_val = 10 if n_estimated >= 10 else 'all'
    return Pipeline([
        ('prep',  pre),
        ('kbest', SelectKBest(score_func=f_classif, k=k_val)),
        ('clf',   estimator)
    ])

X_all = df[sum(feature_sets.values(), [])]
y = df['label']
X_tr_full, X_te_full, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.20, random_state=42, stratify=y)

def slice_split(cols):
    return X_tr_full[cols], X_te_full[cols]

models = {
    'k-NN': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
    'SVC' : SVC(kernel='rbf', shrinking=True, probability=False),
    'ANN' : MLPClassifier(hidden_layer_sizes=(10,15),
                          activation='relu',
                          solver='adam',
                          learning_rate_init=0.001,
                          max_iter=200,
                          random_state=42)
}

results = {}
outer_loop = list(models.items())

for i, (model_name, estimator) in enumerate(outer_loop, start=1):
    results[model_name] = {}
    print("\nModel:", model_name)
    for j, (tag, cols) in enumerate(feature_sets.items(), start=1):
        print("  Feature-set:", tag, flush=True)
        Xtr, Xte = slice_split(cols)
        pipe = make_pipeline(
            cat_cols=[c for c in cols if c in demographic_cols],
            num_cols=[n for n in cols if n in engagement_cols + performance_cols],
            estimator=estimator
        )
        pipe.fit(Xtr, y_tr)
        y_pred = pipe.predict(Xte)
        results[model_name][tag] = {
            'accuracy': round(accuracy_score(y_te, y_pred), 4),
            'f1'      : round(f1_score(y_te, y_pred), 4),
            'jaccard' : round(jaccard_score(y_te, y_pred), 4)
        }

for model, tbl in results.items():
    print("\n===", model, "===")
    display(pd.DataFrame(tbl).T)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




