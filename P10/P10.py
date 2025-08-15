#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# P10 Reproduction Script
"""
Title: "A Withdrawal Prediction Model of At-Risk Learners"
Authors: David Tait, Stephen Lonn, Christopher Brooks
Conference: Proceedings of the 9th International Learning Analytics & Knowledge Conference (LAK 2019)

Purpose:
End-to-end reproduction of the Paper 10 withdrawal prediction pipeline:
    - Data loading and preprocessing from OULAD
    - Feature engineering (demographics, VLE behaviour, assessments)
    - Discretisation of numeric indicators
    - Model training and evaluation across:
        • Decision Tree (J48 equivalent)
        • Random Forest
        • TAN Bayesian Classifier
        • SVM
        • MLP
    - Performance comparison across balanced/unbalanced datasets
"""


# In[ ]:


#Step 1
import os, random, numpy as np, pandas as pd
os.environ["PYTHONHASHSEED"] = "1"
random.seed(1)
np.random.seed(1)


# In[ ]:


#Step 2
student_info = pd.read_csv("studentInfo.csv")
student_reg = pd.read_csv("studentRegistration.csv")
student_assessment = pd.read_csv("studentAssessment.csv")
assessments = pd.read_csv("assessments.csv")
student_vle = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")
courses = pd.read_csv("courses.csv")
MOD, PRES = "DDD", "2013B"
student_info_f = student_info.query("code_module==@MOD and code_presentation==@PRES")
student_reg_f = student_reg.query("code_module==@MOD and code_presentation==@PRES")
student_vle_f = student_vle.query("code_module==@MOD and code_presentation==@PRES")
vle_f = vle.query("code_module==@MOD and code_presentation==@PRES")
courses_f = courses.query("code_module==@MOD and code_presentation==@PRES")
assessments_f = assessments.query("code_module==@MOD and code_presentation==@PRES")
student_assessment_f = student_assessment.merge(
    assessments_f[["id_assessment","code_module","code_presentation"]],
    on="id_assessment", how="inner"
)
df_master = student_info_f.merge(
    student_reg_f,
    on=["code_module","code_presentation","id_student"], how="left"
).merge(
    courses_f,
    on=["code_module","code_presentation"], how="left"
)
assert len(df_master)==1303
print("STEP 2 OK:", df_master.shape)


# In[ ]:


#Step 3
age_map = {"0-35": "young", "35-55": "middle", "55<=": "senior"}
df_demo = df_master[["id_student", "gender", "region", "highest_education",
                     "num_of_prev_attempts", "disability", "age_band"]] \
            .assign(age_group=lambda d: d["age_band"].map(age_map)) \
            .drop(columns="age_band").set_index("id_student")
df_vle_long = student_vle_f.merge(vle_f, on=["code_module","code_presentation","id_site"], how="left")
autonomy = df_vle_long.groupby("id_student")["id_site"].nunique().rename("autonomy")
motivation = df_vle_long.groupby("id_student")["sum_click"].sum().rename("motivation")
bucket_map = {
    "collab": {"forumng","ouwiki","oucollaborate","ouelluminate"},
    "structure": {"homepage","glossary","dataplus","oudictionary","oucontentindex","oucollaborate"},
    "content": {"resource","oucontent","page","subpage","url"},
    "evaluate": {"quiz","questionnaire","ouexam","ouelluminatequiz"}
}
def bucketize(act):
    for label, toys in bucket_map.items():
        if act in toys:
            return label
    return None
df_vle_long["commit_bucket"] = df_vle_long["activity_type"].map(bucketize)
commit_buckets = ["collab","structure","content","evaluate"]
commit = df_vle_long.pivot_table(
    index="id_student", columns="commit_bucket", values="sum_click",
    aggfunc="sum", fill_value=0
).reindex(columns=commit_buckets, fill_value=0).rename(columns={
    "collab":"commit_collab",
    "structure":"commit_structure",
    "content":"commit_content",
    "evaluate":"commit_evaluate"
})
df_assess_long = student_assessment_f.merge(
    assessments_f, on=["code_module","code_presentation","id_assessment"], how="left"
)
df_assess_long["weighted_score"] = df_assess_long["score"] * df_assess_long["weight"]
performance = df_assess_long.groupby("id_student")["weighted_score"].sum().rename("performance")
total_assessments = assessments_f["id_assessment"].nunique()
df_assess_long["on_time"] = df_assess_long["date_submitted"] <= df_assess_long["date"]
perseverance = df_assess_long.groupby("id_student")["on_time"].sum().div(total_assessments).rename("perseverance")
df_indicators = df_demo.join([autonomy, motivation, commit, performance, perseverance]).reset_index()
num_cols = df_indicators.select_dtypes(include="number").columns
df_indicators[num_cols] = df_indicators[num_cols].fillna(0)
print("df_indicators shape:", df_indicators.shape)


# In[ ]:


#Step 4
target = df_master[["id_student", "final_result"]].set_index("id_student") \
    .squeeze().map(lambda x: 0 if x == "Withdrawn" else 1).rename("target")
df_model = df_indicators.set_index("id_student").join(target).reset_index()
print("Target distribution:\n", df_model["target"].value_counts(), "\n")
print("df_model shape:", df_model.shape)


# In[ ]:


#Step 5
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
X = df_model.drop(columns=["id_student","target"])
y = df_model["target"]
cat_cols = ["gender", "region", "highest_education", "age_group", "disability"]
num_cols = [c for c in X.columns if c not in cat_cols]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])
smote_pipe = Pipeline([("pre", preprocess), ("smote", SMOTE(random_state=1))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
train_idx, test_idx = next(cv.split(X, y))
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
print("Before SMOTE:", np.bincount(y_train))
X_res, y_res = smote_pipe.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_res))


# In[ ]:


#Step 6
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
df_disc = df_model.copy()
num_inds = ["autonomy", "motivation", "commit_collab", "commit_structure",
            "commit_content", "commit_evaluate", "performance", "perseverance"]
def discretise(series, rs=1, cutoff=0.15):
    if series.nunique(dropna=False) <= 1:
        return pd.Series(["low"] * len(series), index=series.index)
    X = series.values.reshape(-1,1)
    km2 = KMeans(n_clusters=2, random_state=rs).fit(X)
    km3 = KMeans(n_clusters=3, random_state=rs).fit(X)
    drop = (km2.inertia_ - km3.inertia_) / km2.inertia_
    if drop >= cutoff:
        labels, mapping = km3.labels_, {0:"low",1:"medium",2:"high"}
    else:
        labels, mapping = km2.labels_, {0:"low",1:"high"}
    return pd.Series(labels, index=series.index).map(mapping)
for col in num_inds:
    df_disc[col] = discretise(df_disc[col])
X_raw = df_disc.drop(columns=["id_student","target"])
y = df_disc["target"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc = enc.fit_transform(X_raw)
print("df_disc shape:", df_disc.shape)
print("Encoded matrix shape:", X_enc.shape)


# In[ ]:


#Step 7

#Step 7.1
df_vle_long = student_vle_f.merge(
    vle_f,
    on=["code_module","code_presentation","id_site"],
    how="left"
)
X_vle_raw = df_vle_long.pivot_table(
    index="id_student",
    columns="activity_type",
    values="sum_click",
    aggfunc="sum",
    fill_value=0
)
demo_cols = [
    "gender",
    "region",
    "highest_education",
    "age_band",
    "num_of_prev_attempts",
    "disability"
]
X_demo = df_master.set_index("id_student")[demo_cols]
X_base = (
    df_model[["id_student"]]
      .set_index("id_student")
      .join(X_demo)
      .join(X_vle_raw)
      .reset_index(drop=True)
)
prep_base = ColumnTransformer([
    ("cat", SKPipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), demo_cols),
    ("num", SKPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), X_vle_raw.columns.tolist())
])

#Step 7.2
X_cont = df_model.drop(columns=["id_student","target"])
prep_cont = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
        ["gender","region","highest_education","age_group","disability"]),
    ("num", StandardScaler(),
        ["autonomy","motivation",
         "commit_collab","commit_structure",
         "commit_content","commit_evaluate",
         "performance","perseverance"])
])

#Step 7.3
X_disc = df_disc.drop(columns=["id_student","target"])
prep_disc = ColumnTransformer([
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        X_disc.columns.tolist())
])

#Step 7.4
y = df_model["target"]

#Step 7.5
variants = [
    ("Unbal / No Ind",   X_base,  prep_base, False),
    ("Unbal / Numeric",  X_cont,  prep_cont, False),
    ("Unbal / Discrete", X_disc,  prep_disc, False),
    ("Bal / No Ind",     X_base,  prep_base, True),
    ("Bal / Numeric",    X_cont,  prep_cont, True),
    ("Bal / Discrete",   X_disc,  prep_disc, True),
]

#Step 7.6
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
dt = DecisionTreeClassifier(
    random_state=1,
    criterion="entropy",
    ccp_alpha=0.0
)
results = []
for label, Xv, prep, balance in variants:
    steps = [("pre", prep)]
    if balance:
        steps.append(("smote", SMOTE(random_state=1)))
    steps.append(("clf", dt))
    pipe = Pipeline(steps)
    f1 = cross_val_score(pipe, Xv, y, scoring="f1", cv=cv, n_jobs=-1)
    results.append((label, f1.mean().round(3)))
df_dt = pd.DataFrame(results, columns=["Variant","F1"]).set_index("Variant")
df_dt = df_dt.style.set_caption("Decision trees")
df_dt

#Step 7.7
rf = RandomForestClassifier(n_estimators=100, random_state=1)
results = []
for label, Xv, prep, balance in variants:
    steps = [("pre", prep)]
    if balance:
        steps.append(("smote", SMOTE(random_state=1)))
    steps.append(("clf", rf))
    pipe = Pipeline(steps)
    f1 = cross_val_score(pipe, Xv, y, scoring="f1", cv=cv, n_jobs=-1)
    results.append((label, f1.mean().round(3)))
df_rf = pd.DataFrame(results, columns=["Variant","F1"]).set_index("Variant")
df_rf = df_rf.style.set_caption("Random Forest")
df_rf

#Step 7.8
def foldwise_binarize(df_train, df_test):
    tr = df_train.copy()
    te = df_test.copy()
    for c in tr.columns:
        if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 2:
            med = tr[c].median()
            tr[c] = np.where(tr[c] <= med, "low", "high")
            te[c] = np.where(te[c] <= med, "low", "high")
        else:
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)
    return tr, te
results = []
for label, Xv, prep_unused, balance in variants:
    f1s = []
    for train_idx, test_idx in cv.split(Xv, y):
        X_tr, X_te = Xv.iloc[train_idx], Xv.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        tr_bin, te_bin = foldwise_binarize(X_tr, X_te)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_enc = enc.fit_transform(tr_bin)
        if balance:
            X_tr_bal, y_tr_bal = SMOTE(random_state=1).fit_resample(X_tr_enc, y_tr.values)
            tr_bal = pd.DataFrame(enc.inverse_transform(X_tr_bal), columns=tr_bin.columns)
            y_bal = pd.Series(y_tr_bal)
        else:
            tr_bal, y_bal = tr_bin, y_tr
        train_df = tr_bal.copy()
        train_df["target"] = y_bal.astype(str)
        test_df = te_bin.copy()
        test_df["target"] = y_te.astype(str)
        ts = TreeSearch(train_df)
        dag = ts.estimate(estimator_type="tan", class_node="target")
        bn = BayesianModel(dag.edges())
        bn.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu")
        y_pred = bn.predict(test_df.drop(columns="target"))["target"].astype(int)
        f1s.append(f1_score(test_df["target"].astype(int), y_pred))
    results.append((label, np.mean(f1s).round(3)))
df_tan = pd.DataFrame(results, columns=["Variant","F1"]).set_index("Variant")
df_tan = df_tan.style.set_caption("Bayesian Classifier (TAN)")
df_tan

#Step 7.9
f1w = make_scorer(f1_score, pos_label=0)
svm = SVC(kernel="linear", C=1, tol=0.1, random_state=1)
dense = FunctionTransformer(
    lambda X: X.A if hasattr(X, "A")
              else (X.toarray() if hasattr(X, "toarray") else X),
    accept_sparse=True
)
results = []
for label, Xv, prep, balance in variants:
    steps = [("pre", prep), ("dense", dense), ("norm", MinMaxScaler())]
    if balance:
        steps.append(("smote", SMOTE(random_state=1)))
    steps.append(("clf", svm))
    f1 = cross_val_score(Pipeline(steps), Xv, y, scoring=f1w, cv=cv, n_jobs=-1)
    results.append((label, np.round(f1.mean(), 3)))
df_svm = pd.DataFrame(results, columns=["Variant", "F1"]).set_index("Variant")
df_svm = df_svm.style.set_caption("Support Vector Machine (SVM)")
df_svm

#Step 7.10
mlp = MLPClassifier(
    solver='sgd',
    learning_rate_init=0.3,
    momentum=0.2,
    random_state=1
)
results = []
for label, Xv, prep, balance in variants:
    steps = [("pre", prep)]
    if balance:
        steps.append(("smote", SMOTE(random_state=1)))
    steps.append(("clf", mlp))
    pipe = Pipeline(steps)
    f1 = cross_val_score(pipe, Xv, y, scoring="f1", cv=cv, n_jobs=-1)
    results.append((label, round(f1.mean(), 3)))
df_mlp = pd.DataFrame(results, columns=["Variant","F1"]).set_index("Variant")
df_mlp.style.set_caption("Artificial Neural Network (MLP)")


# In[ ]:


#Step 8

def _unwrap(styler_or_df):
    return styler_or_df.data if hasattr(styler_or_df, "data") else styler_or_df

frames = []
for name, df_model in [
        ("Decision Tree",  df_dt),
        ("Random Forest",  df_rf),
        ("SVM",            df_svm),
        ("MLP",            df_mlp),
        ("Bayesian (TAN)", df_tan)]:
    tmp = _unwrap(df_model).reset_index()
    tmp["Model"] = name
    frames.append(tmp[["Model", "Variant", "F1"]])

df_all = pd.concat(frames, ignore_index=True)

order = ["Unbal / No Ind", "Unbal / Numeric", "Unbal / Discrete",
         "Bal / No Ind",   "Bal / Numeric",   "Bal / Discrete"]

(df_all.assign(F1=lambda d: d["F1"].round(3))
      .pivot(index="Model", columns="Variant", values="F1")
      .reindex(columns=order)
      .style.set_caption("F1 scores"))

