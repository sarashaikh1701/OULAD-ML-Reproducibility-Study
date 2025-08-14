#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Paper 5 Reproduction Script
"""
Title: "Predicting at-risk university students in a virtual learning environment via a machine learning algorithm"
Authors: Kwok Tai Chui, Dennis Chun Lok Fung, Miltiadis D. Lytras, Tin Miu Lam (Computers in Human Behavior, 2020)



Purpose:
End-to-end entry-point script for RTV-SVM reproduction with exact-match settings:
    - Data loading and filtering (exclude Withdrawn; labels from final_score)
    - Feature engineering: 10 numeric + k-1 one-hot for categorical = 52 predictors
    - Preprocessing: mean impute + z-scale (numeric), k-1 OneHotEncoder (categorical)
    - Five-fold Stratified CV per module
    - RTV-SVM reductions: tier-1 (log-pdf) and tier-2 (projection) scenarios S1–S4
    - Metrics: R_tv, M_r, T1, T2, Se, Sp, OA
    - Outputs: printed aggregates for Tables 3, 4, and 5

Notes:
    - RNG seeds are set at the top of the script.
    - No logic added or removed from the original implementation.
"""


# In[ ]:


# Step 1: Seeds and imports
import os, random, time, warnings, numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import multivariate_normal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


def main():
    # Step 1: Load OULAD CSVs
    student_info        = pd.read_csv("studentInfo.csv")
    student_vle         = pd.read_csv("studentVle.csv")
    student_assessment  = pd.read_csv("studentAssessment.csv")
    assessments         = pd.read_csv("assessments.csv")
    student_registration= pd.read_csv("studentRegistration.csv")

    # Step 2: Labels
    student_info = student_info[student_info.final_result != "Withdrawn"].copy()
    merged = (student_assessment
              .merge(assessments[['id_assessment','weight',
                                  'code_module','code_presentation']],
                     on='id_assessment', how='left'))
    merged['weighted'] = merged['score'] * merged['weight'] / 100.0
    final_scores = (merged
                    .groupby(['id_student','code_module','code_presentation'],
                             as_index=False)['weighted']
                    .sum()
                    .rename(columns={'weighted':'final_score'}))
    student_info = (student_info
                    .merge(final_scores, on=['id_student','code_module',
                                             'code_presentation'], how='left'))
    student_info = student_info[student_info.final_score.notna()]
    student_info['label_binary'] = (student_info.final_score < 40).astype(int)
    def _tri(s):
        if s >= 55:   return 0
        if s >= 40:   return 1
        return 2
    student_info['label_multi'] = student_info.final_score.apply(_tri)

    # Step 3: Activity and assessment features
    vle_feat = (
        student_vle
        .groupby(["id_student", "code_module", "code_presentation"])
        .agg(total_clicks   = ("sum_click", "sum"),
             days_active    = ("date",      "nunique"),
             resources_used = ("id_site",   "nunique"))
        .reset_index()
    )
    sa_feat = (
        student_assessment
        .merge(assessments[["id_assessment", "weight",
                            "code_module", "code_presentation"]],
               on="id_assessment", how="left")
        .groupby(["id_student", "code_module", "code_presentation"])
        .agg(n_assess   = ("id_assessment", "size"),
             mean_score = ("score", "mean"),
             std_score  = ("score", "std"),
             miss_asmt  = ("score", lambda s: s.isna().sum()))
        .reset_index()
    )

    # Step 4: Master table
    master = (
        student_info
        .merge(student_registration[["id_student","code_module",
                                     "code_presentation","date_registration"]],
               on=["id_student","code_module","code_presentation"], how="left")
        .merge(vle_feat, on=["id_student","code_module","code_presentation"],
               how="left")
        .merge(sa_feat, on=["id_student","code_module","code_presentation"],
               how="left")
    )
    behav_cols = ["total_clicks","days_active","resources_used",
                  "n_assess","mean_score","std_score","miss_asmt"]
    master[behav_cols] = master[behav_cols].fillna(0)
    bins   = [0,30,60,90,120,np.inf]
    labels = ["30","60","90","120","150+"]
    master["studied_credits_cat"] = pd.cut(master.studied_credits,
                                           bins=bins, labels=labels, right=False)

    # Step 5: Explicit category sets
    cat_levels = {
        "code_module": ["AAA","BBB","CCC","DDD","EEE","FFF","GGG"],
        "code_presentation": ["2013B","2013J","2014B","2014J"],
        "gender": ["F","M"],
        "region": [
            "East Anglian Region","East Midlands Region","Ireland","London Region",
            "North Region","North Western Region","Northern Ireland","Scotland",
            "South East Region","South Region","South West Region","Wales",
            "West Midlands Region"
        ],
        "highest_education": [
            "No Formal quals","Lower Than A Level","A Level or Equivalent",
            "HE Qualification","Post Graduate Qualification"
        ],
        "imd_band": [
            "0-10%","10-20%","20-30%","30-40%","40-50%",
            "50-60%","60-70%","70-80%","80-90%","90-100%"
        ],
        "age_band": ["0-35","35-55","55<="],
        "disability": ["N","Y"],
        "studied_credits_cat": labels,
    }
    for col, cats in cat_levels.items():
        master[col] = pd.Categorical(master[col], categories=cats)

    # Step 6: Predictor lists
    numeric_cols = [
        "studied_credits","num_of_prev_attempts","date_registration",
        *behav_cols,
    ]
    categorical_cols = list(cat_levels.keys())

    # Step 7: Preprocessing
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(categories=[cat_levels[c] for c in categorical_cols],
                                   drop="first", sparse=True))
        ]), categorical_cols)
    ])
    X_mat = preproc.fit_transform(master)
    y_bin = master["label_binary"].values
    y_tri = master["label_multi"].values

    # Step 8: Sanity check
    ohe = preproc.named_transformers_["cat"]["ohe"]
    feat_names = list(numeric_cols) + list(ohe.get_feature_names(categorical_cols))
    assert X_mat.shape[1] == 52, f"{X_mat.shape[1]} ≠ 52 predictors"
    X_df = pd.DataFrame(X_mat.toarray() if sparse.issparse(X_mat) else X_mat,
                        columns=feat_names, index=master.index)
    X = X_df.values
    print("✔ predictor count:", X.shape[1])

    # Step 9: RTV-SVM helpers
    R_TH = 0.35
    def tier1_pdf_mask(X, y):
        keep = np.zeros_like(y, dtype=bool)
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            Xc  = X[idx]
            mu  = Xc.mean(0)
            Σ   = np.diag(Xc.var(0) + 1e-6)
            logp = multivariate_normal.logpdf(Xc, mu, Σ)
            r    = np.exp(logp.min() - logp.max())
            Nt   = max(int(np.floor(len(Xc) * r / R_TH)), 1)
            keep[idx[np.argsort(logp)[:Nt]]] = True
        return keep
    def tier2_proj_mask(X, y, init):
        keep = init.copy()
        while True:
            changed = False
            I = np.where(keep)[0]
            Xk, yk = X[I], y[I]
            mu0, mu1 = Xk[yk==0].mean(0), Xk[yk==1].mean(0)
            v = mu1 - mu0
            v_len = np.linalg.norm(v) + 1e-12
            for cls, mu_s in ((0, mu0), (1, mu1)):
                Ic = I[yk == cls]
                if Ic.size <= 1:
                    continue
                d    = X[Ic] - mu_s
                dlen = np.linalg.norm(d, axis=1) + 1e-12
                cos  = (d @ v) / (dlen * v_len)
                proj = dlen * cos
                forward = cos > 0
                if not forward.any():
                    continue
                kill = Ic[forward][proj[forward].argmin()]
                keep[kill] = False
                changed = True
            if not changed:
                break
        return keep
    def _svc():
        return SVC(kernel="rbf", C=10.0, gamma="auto")
    def svm_metrics(Xtr, ytr, Xte, yte):
        clf = _svc(); t0=time.time(); clf.fit(Xtr, ytr);  T1=time.time()-t0
        t0=time.time(); pred = clf.predict(Xte);          T2=time.time()-t0
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        Se, Sp = tp/(tp+fn+1e-12), tn/(tn+fp+1e-12)
        OA      = accuracy_score(yte, pred)
        return dict(T1=T1, T2=T2, Se=Se, Sp=Sp, OA=OA), clf
    def run_scenarios(Xtr, ytr, Xte, yte):
        base, ref = svm_metrics(Xtr, ytr, Xte, yte)
        sv_mask = np.zeros_like(ytr, bool); sv_mask[ref.support_] = True
        out = {"S1": dict(R_tv=0.0, M_r="N/A", **base)}
        k1 = tier1_pdf_mask(Xtr, ytr) | sv_mask
        met,_ = svm_metrics(Xtr[k1], ytr[k1], Xte, yte)
        out["S2"] = dict(R_tv=100*(1-k1.mean()), M_r="No", **met)
        k2 = tier2_proj_mask(Xtr, ytr, np.ones_like(ytr, bool)) | sv_mask
        met,_ = svm_metrics(Xtr[k2], ytr[k2], Xte, yte)
        out["S3"] = dict(R_tv=100*(1-k2.mean()), M_r="No", **met)
        k12 = tier2_proj_mask(Xtr, ytr, k1) | sv_mask
        met,_ = svm_metrics(Xtr[k12], ytr[k12], Xte, yte)
        out["S4"] = dict(R_tv=100*(1-k12.mean()), M_r="No", **met)
        return out

        # Step 10: Re-create Tables 3–5
    wanted = ["AAA","BBB","CCC","DDD","EEE","FFF","GGG"]
    order  = ["S1","S2","S3","S4"]
    cv     = StratifiedKFold(5, shuffle=True, random_state=0)
    rows3, rows5 = [], []

    for mod in wanted:
        m = master.code_module == mod
        Xc = X[m]
        y_bin_c  = y_bin[m]
        y_both_c = np.where(y_tri[m] == 0, 0, 1)

        if np.unique(y_bin_c).size < 2:
            continue

        for tr, te in cv.split(Xc, y_bin_c):
            for scen, met in run_scenarios(Xc[tr], y_bin_c[tr], Xc[te], y_bin_c[te]).items():
                rows3.append(dict(module=mod, scenario=scen, **met))

        if np.unique(y_both_c).size < 2:
            continue

        for tr, te in cv.split(Xc, y_both_c):
            for scen, met in run_scenarios(Xc[tr], y_both_c[tr], Xc[te], y_both_c[te]).items():
                rows5.append(dict(module=mod, scenario=scen, **met))

    # Step 11: Extra counts (single print after loop)
    cnt = (student_info.groupby(["code_module", "final_result"]).size().unstack(fill_value=0))
    print(cnt.loc["GGG"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




