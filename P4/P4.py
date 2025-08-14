#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1 Setup
import os
import time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Masking
from tensorflow.keras.optimizers import Adam

np.random.seed(42)


# In[ ]:


# Step 2 + 3 Data Manipulation + Feature Engineering
si = pd.read_csv("studentInfo.csv")
sv = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")
courses = pd.read_csv("courses.csv")
print("Loaded studentInfo.csv:", si.shape)
print("Loaded studentVle.csv:", sv.shape)
print("Loaded vle.csv:", vle.shape)

si = si[si.final_result != "Withdrawn"].copy()
si["final_result"] = si["final_result"].replace({"Distinction": "Pass"})
si["student_course_id"] = (
    si["id_student"].astype(str) + "_" +
    si["code_module"] + "_" + si["code_presentation"]
)
print("Cleaned studentInfo:", si.shape)

sv = (
    sv
    .merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    .merge(
        si[["id_student", "code_module", "code_presentation", "final_result"]],
        on=["id_student", "code_module", "code_presentation"],
        how="inner"
    )
)
sv["week"] = (sv["date"] // 7) + 1

weekly = (
    sv.groupby(
        ["id_student", "code_module", "code_presentation", "week", "activity_type"],
        as_index=False
    ).agg({"sum_click": "sum"})
)

pivot = weekly.pivot_table(
    index=["id_student", "code_module", "code_presentation", "week"],
    columns="activity_type",
    values="sum_click",
    fill_value=0
).reset_index()

expected_activities = [
    'dataplus', 'dualpane', 'externalquiz', 'folder', 'forumng', 'glossary',
    'homepage', 'htmlactivity', 'oucollaborate', 'oucontent', 'ouelluminate',
    'ouwiki', 'page', 'questionnaire', 'quiz', 'repeatactivity', 'resource',
    'sharedsubpage', 'subpage', 'url'
]

for act in expected_activities:
    if act not in pivot.columns:
        pivot[act] = 0

pivot = pivot[["id_student", "code_module", "code_presentation", "week"] + expected_activities]

X_list, y_list = [], []
for (stu, mod, pres), group in pivot.groupby(["id_student", "code_module", "code_presentation"]):
    group = group.set_index("week").reindex(range(1, 39), fill_value=0)
    X_list.append(group[expected_activities].values)
    label = si.query(
        "id_student == @stu and code_module == @mod and code_presentation == @pres"
    )["final_result"].iat[0]
    y_list.append(1 if label == "Pass" else 0)

X = np.stack(X_list)
y = np.array(y_list)

print("Final shape of X:", X.shape)
print("Final shape of y:", y.shape)
print(si["id_student"].nunique())
print(si.shape)
print(sv["id_student"].nunique())


# In[ ]:


# Step 4 LSTM Model
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("history_logs", exist_ok=True)

results = []

for week in [5, 10, 20, 38]:
    model_path = f"models/lstm_week{week}.h5"
    y_pred_path = f"predictions/y_pred_week{week}.npy"
    y_true_path = f"predictions/y_true_week{week}.npy"
    history_path = f"history_logs/history_week{week}.npy"

    print(f"\n=== Week {week} ===")
    if os.path.exists(model_path):
        print(f"Model for week {week} already exists. Skipping training.")
        model = load_model(model_path)
        y_pred = np.load(y_pred_path)
        y_test = np.load(y_true_path)
    else:
        print(f"Training model for week {week}...")
        X_week = X[:, :week, :]
        X_padded = np.zeros((X.shape[0], 38, X.shape[2]))
        X_padded[:, :week, :] = X_week

        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y, test_size=0.2, stratify=y, random_state=42
        )

        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(38, 20)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(300))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_train, y_train,
            epochs=60,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        model.save(model_path)
        np.save(history_path, history.history)

        y_pred = model.predict(X_test).flatten()
        y_pred_labels = (y_pred >= 0.5).astype(int)

        np.save(y_pred_path, y_pred_labels)
        np.save(y_true_path, y_test)

    y_pred_labels = (y_pred >= 0.5).astype(int)
    report = classification_report(y_test, y_pred_labels, output_dict=True, zero_division=0)
    results.append({
        'week': week,
        'accuracy': round(report['accuracy'], 4),
        'precision': round(report['1']['precision'], 4),
        'recall': round(report['1']['recall'], 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("lstm_weekwise_results.csv", index=False)
print("\n Training complete. Results saved to lstm_weekwise_results.csv.")


# In[ ]:


# Step 5 Validate Saved Model File
with h5py.File(model_path, 'r') as f:
    print("Model file keys:", list(f.keys()))


# In[ ]:


# Step 6 Reload Predictions
results = []
for week in [5, 10, 20, 38]:
    print(f"\n=== Reloading Week {week} Predictions ===")
    y_pred_path = f"predictions/y_pred_week{week}.npy"
    y_true_path = f"predictions/y_true_week{week}.npy"
    if os.path.exists(y_pred_path) and os.path.exists(y_true_path):
        y_pred = np.load(y_pred_path)
        y_test = np.load(y_true_path)
        y_pred_labels = (y_pred >= 0.5).astype(int)
        report = classification_report(y_test, y_pred_labels, output_dict=True, zero_division=0)
        results.append({
            'week': week,
            'accuracy': round(report['accuracy'], 4),
            'precision': round(report['1']['precision'], 4),
            'recall': round(report['1']['recall'], 4)
        })
    else:
        print(f"Missing files for week {week}, skipping.")

results_df = pd.DataFrame(results)
results_df.to_csv("lstm_weekwise_results.csv", index=False)
print("\n Reloaded results saved to lstm_weekwise_results.csv.")


# In[ ]:


# Step 7 Baseline Models
baseline_results = []
for week in tqdm([5, 10, 20, 38], desc="Retrying Weeks"):
    print(f"\n--- Baseline Evaluation for Week {week} ---")
    X_week = X[:, :week, :].sum(axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_week, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', max_iter=1000, verbose=1),
        "SVM": LinearSVC(max_iter=1000, verbose=1),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=200,
            early_stopping=True,
            random_state=42,
            verbose=True
        )
    }
    for model_name in tqdm(models.keys(), desc=f"Training Models for Week {week}"):
        model = models[model_name]
        print(f"\nTraining {model_name}...")
        if hasattr(model, "solver"):
            print(f"Using solver for {model_name}: {model.solver}")
        start = time.time()
        model.fit(X_train, y_train)
        runtime = round(time.time() - start, 2)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = round(report["accuracy"], 4)
        prec = round(report["1"]["precision"], 4)
        rec = round(report["1"]["recall"], 4)
        print(f"{model_name} @ Week {week} → Accuracy: {acc}, Precision: {prec}, Recall: {rec} (Time: {runtime}s)")
        baseline_results.append({
            "model": model_name,
            "week": week,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "runtime_sec": runtime
        })

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv("baseline_model_results.csv", mode='a', index=False, header=False)
print("\n Baseline results appended to baseline_model_results.csv")
baseline_df.drop_duplicates(subset=["model", "week"], keep='last', inplace=True)
baseline_df.to_csv("baseline_model_results.csv", index=False)


# In[ ]:




