#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# P7 Reproduction Script 
"""
Title: "Joint RNN Models for Early Prediction of Student Performance in Online Learning"
Authors: He, Xiaoxiao; Tang, Jiliang; et al. (2020)

Purpose:
End-to-end entry-point script for reproducing joint RNN-based early at-risk prediction:
    - Environment and RNG seeding
    - Data loading, preprocessing, and tensor assembly
    - Baseline RNN and Joint GRU model definitions
    - Cyclic training with per-course splits, CSV logs, and weights
    - Overall and week-by-week evaluation metrics
"""


# In[ ]:


# Step 1 — Environment & RNG
from datetime import datetime
from tensorflow.python.client import device_lib
import numpy as np, pandas as pd, tensorflow as tf, random, os, h5py, gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Concatenate, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras import backend as K

print("TensorFlow build :", tf.__version__)
print("CUDA enabled     :", tf.test.is_built_with_cuda())
print("GPU available    :", tf.test.is_gpu_available())

local_devices = device_lib.list_local_devices()
gpu_devices = [x.name for x in local_devices if x.device_type == 'GPU']
print("Visible GPUs     :", gpu_devices)

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

print("Session started  :", datetime.now())

import tensorflow as tf
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())


# In[ ]:


# Step 2 — Utilities & Training Config
import os, h5py, numpy as np

def _bytes(x):
    return np.string_(x) if not isinstance(x, bytes) else x

def _fix_attrs(h5obj):
    for key in list(h5obj.attrs):
        val = h5obj.attrs[key]
        if isinstance(val, str):
            del h5obj.attrs[key]
            h5obj.attrs.create(key, _bytes(val))
        elif isinstance(val, np.ndarray) and val.dtype.kind in ('U', 'O'):
            del h5obj.attrs[key]
            h5obj.attrs.create(key, np.array([_bytes(v) for v in val], dtype='S'))
    for child in h5obj.values():
        if isinstance(child, h5py.Group):
            _fix_attrs(child)

def safe_load_weights(model, path):
    if not (path and os.path.exists(path) and h5py.is_hdf5(path)):
        return
    with h5py.File(path, 'r+') as f:
        _fix_attrs(f)
    model.load_weights(path)

TRAINING_CONFIG = {
    "batch_size": 256,
    "epochs": 250,
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "early_week": 5,
    "last_week": 39,
}

print("== Training configuration ==")
print(TRAINING_CONFIG)


# In[ ]:


# Step 3 — Data Loading & Preprocessing
# Step 3.1 — Read CSVs
import os
import pandas as pd
import numpy as np

DATA_DIR = '.'
WEEKS   = 40

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found – double-check the folder.")
    return pd.read_csv(path)

raw = {
    "assessments"       : load_csv("assessments.csv"),
    "courses"           : load_csv("courses.csv"),
    "studentAssessment" : load_csv("studentAssessment.csv"),
    "studentInfo"       : load_csv("studentInfo.csv"),
    "studentRegistration": load_csv("studentRegistration.csv"),
    "studentVle"        : load_csv("studentVle.csv"),
    "vle"               : load_csv("vle.csv"),
}
for k, df in raw.items():
    print(f"{k:<20} → {df.shape}")

# Step 3.2 — Target & Demographics
stu_info = raw["studentInfo"].copy()
stu_info = stu_info[stu_info["final_result"] != "Withdrawn"]
PASS_LABELS = {"Pass", "Distinction"}
stu_info["label"] = np.where(stu_info["final_result"].isin(PASS_LABELS), 1, 0)
print(stu_info["label"].value_counts(dropna=False))

DEMOG_COLS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
demog_onehot = pd.get_dummies(stu_info[DEMOG_COLS], prefix=DEMOG_COLS)
print(f"Demographic matrix shape  {demog_onehot.shape}")

# Step 3.3 — Weekly Assessment Matrix
sa = (
    raw["studentAssessment"]
    .merge(
        raw["assessments"][["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment", how="left"
    )
)
sa["week"] = (sa["date_submitted"] // 7).clip(lower=0, upper=WEEKS-1)
ass_stream = (
    sa.groupby(["id_student", "code_module", "code_presentation", "week"])["score"]
      .sum()
      .reset_index()
      .rename(columns={"score": "score_sum"})
)
ass_matrix = (
    ass_stream
    .pivot_table(index=["id_student", "code_module", "code_presentation"],
                 columns="week", values="score_sum", fill_value=0)
    .reindex(columns=range(WEEKS), fill_value=0)
)
print("Assessment matrix shape:", ass_matrix.shape)

# Step 3.4 — Weekly Click Matrix
sv = raw["studentVle"].copy()
sv["week"] = (sv["date"] // 7).clip(lower=0, upper=WEEKS-1)
click_stream = (
    sv.groupby(["id_student", "code_module", "code_presentation", "week"])["sum_click"]
      .sum()
      .reset_index()
      .rename(columns={"sum_click": "clicks"})
)
click_matrix = (
    click_stream
    .pivot_table(index=["id_student", "code_module", "code_presentation"],
                 columns="week", values="clicks", fill_value=0)
    .reindex(columns=range(WEEKS), fill_value=0)
)
print("Click matrix shape:", click_matrix.shape)

# Step 3.5 — Final Design Matrix
if not str(ass_matrix.columns[0]).startswith("ass_"):
    ass_matrix.columns = [f"ass_{wk}" for wk in ass_matrix.columns]
if not str(click_matrix.columns[0]).startswith("click_"):
    click_matrix.columns = [f"click_{wk}" for wk in click_matrix.columns]
ASS_COLS   = list(ass_matrix.columns)
CLICK_COLS = list(click_matrix.columns)
base_df = stu_info.set_index(["id_student", "code_module", "code_presentation"])
demog_onehot.index = base_df.index
merged = (
    base_df
    .join(demog_onehot, how="left")
    .join(ass_matrix,    how="left")
    .join(click_matrix,  how="left")
)
merged[ASS_COLS + CLICK_COLS] = merged[ASS_COLS + CLICK_COLS].fillna(0)
print("Final design-matrix shape:", merged.shape)

TARGET = merged["label"].values
DEMOG  = merged[demog_onehot.columns].values
ASSESS = merged[ASS_COLS].values.reshape(-1, WEEKS, 1)
CLICKS = merged[CLICK_COLS].values.reshape(-1, WEEKS, 1)

print("DEMOG :", DEMOG.shape,
      "\nASSESS:", ASSESS.shape,
      "\nCLICKS:", CLICKS.shape,
      "\nLABEL :", TARGET.shape)


# In[ ]:


# Step 4 — Model & Training
# Step 4.1 — Shapes & Split Helper
import numpy as np, pandas as pd, tensorflow as tf, random, os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, SimpleRNN, GRU, LSTM, Concatenate, LeakyReLU, Dropout)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

N_STUDENTS = TARGET.shape[0]
DEMOG_DIM  = DEMOG.shape[1]
TIME_STEPS = ASSESS.shape[1]
SEQ_FEATS  = 1

print("N_STUDENTS :", N_STUDENTS, "\nDEMOG_DIM :", DEMOG_DIM, "\nTIME_STEPS:", TIME_STEPS)

def make_course_split(current_mod, current_pres, val_ratio=0.20):
    mask_prev = (
        (merged.index.get_level_values("code_module") <  current_mod) |
        ((merged.index.get_level_values("code_module") == current_mod) &
         (merged.index.get_level_values("code_presentation") < current_pres))
    )
    prev_idx = np.where(mask_prev)[0]
    if prev_idx.size == 0:
        raise ValueError("No historical courses before "
                         f"{current_mod} {current_pres} to train on.")
    rng = np.random.RandomState(SEED)
    rng.shuffle(prev_idx)
    split = int((1 - val_ratio) * prev_idx.size)
    train_idx, val_idx = prev_idx[:split], prev_idx[split:]
    mask_test = (
        (merged.index.get_level_values("code_module") == current_mod) &
        (merged.index.get_level_values("code_presentation") == current_pres)
    )
    test_idx = np.where(mask_test)[0]
    return train_idx, val_idx, test_idx

# Step 4.2 — Baseline Builders
def build_baseline(rnn_type="GRU",
                   demog_dim=DEMOG_DIM,
                   time_steps=TIME_STEPS,
                   seq_feats=SEQ_FEATS,
                   lr=TRAINING_CONFIG["learning_rate"]):
    inp_demog  = Input(shape=(demog_dim,),  name="demographics")
    inp_assess = Input(shape=(time_steps, seq_feats), name="ass_seq")
    inp_click  = Input(shape=(time_steps, seq_feats), name="click_seq")

    x_dem = Dense(128)(inp_demog); x_dem = LeakyReLU()(x_dem)
    x_dem = Dense(128)(x_dem);     x_dem = LeakyReLU()(x_dem)

    RNNLayer = {"SimpleRNN": SimpleRNN, "GRU": GRU, "LSTM": LSTM}[rnn_type]

    x_ass = inp_assess
    for i in range(7):
        return_seq = i < 6
        x_ass = RNNLayer(256, return_sequences=return_seq, name=f"ass_{rnn_type}_{i+1}")(x_ass)

    x_clk = inp_click
    for i in range(7):
        return_seq = i < 6
        x_clk = RNNLayer(256, return_sequences=return_seq, name=f"clk_{rnn_type}_{i+1}")(x_clk)

    concat = Concatenate()([x_dem, x_ass, x_clk])
    y = Dense(384)(concat); y = LeakyReLU()(y)
    y = Dense(768)(y);      y = LeakyReLU()(y)
    y = Dense(1536)(y);     y = LeakyReLU()(y)
    out = Dense(1, activation="sigmoid")(y)

    model = Model([inp_demog, inp_assess, inp_click], out, name=f"Baseline_{rnn_type}")
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

gru_baseline = build_baseline("GRU")
gru_baseline.summary(line_length=120)

def build_joint_rnn_gru(demog_dim=DEMOG_DIM, time_steps=TIME_STEPS, seq_feats=SEQ_FEATS, lr=TRAINING_CONFIG["learning_rate"]):
    from keras.layers import Input, Dense, GRU, Concatenate, LeakyReLU
    from keras.models import Model
    from keras.optimizers import Adam
    inp_demog  = Input(shape=(demog_dim,),            name="demographics")
    inp_assess = Input(shape=(time_steps, seq_feats), name="ass_seq")
    inp_click  = Input(shape=(time_steps, seq_feats), name="click_seq")
    x_dem = Dense(128)(inp_demog); x_dem = LeakyReLU()(x_dem)
    x_dem = Dense(128)(x_dem);     x_dem = LeakyReLU()(x_dem)
    shared = [GRU(256, return_sequences=(i < 6), name=f"shared_GRU_{i+1}") for i in range(7)]
    def run_stack(x):
        for layer in shared:
            x = layer(x)
        return x
    x_ass = run_stack(inp_assess)
    x_clk = run_stack(inp_click)
    concat = Concatenate()([x_dem, x_ass, x_clk])
    y = Dense(384)(concat);  y = LeakyReLU()(y)
    y = Dense(768)(y);       y = LeakyReLU()(y)
    y = Dense(1536)(y);      y = LeakyReLU()(y)
    out = Dense(1, activation="sigmoid")(y)
    model = Model([inp_demog, inp_assess, inp_click], out, name="Joint_RNN_GRU")
    model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Step 4.3 — Cyclic Training, CSV Logs, Weights
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py, os, gc
from tqdm import tqdm

EPOCHS_TOTAL = TRAINING_CONFIG["epochs"]
CYCLE        = 50
BATCH_SAFE   = 128

SCHEDULE = [
    ("AAA", "2014J"),
    ("BBB", "2014B"), ("BBB", "2014J"),
    ("CCC", "2014J"),
    ("DDD", "2014B"), ("DDD", "2014J"),
    ("EEE", "2014B"), ("EEE", "2014J"),
    ("FFF", "2014B"), ("FFF", "2014J"),
    ("GGG", "2014B"), ("GGG", "2014J"),
]
results = []

def last_completed_epoch(csv_path):
    if not os.path.exists(csv_path):
        return -1
    import pandas as pd
    try:
        return pd.read_csv(csv_path)["epoch"].max()
    except Exception:
        return -1

for mod, pres in tqdm(SCHEDULE, desc="Test courses", unit="course"):
    tr_idx, val_idx, te_idx = make_course_split(mod, pres)
    best_path = f"{mod}_{pres}_best.weights.h5"
    csv_path  = f"{mod}_{pres}.csv"
    ckpt_save  = ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    csv_logger = CSVLogger(csv_path, append=True)
    last_epoch = last_completed_epoch(csv_path)
    START_FROM = max(0, (last_epoch + 1))
    if START_FROM % CYCLE:
        START_FROM = (START_FROM // CYCLE) * CYCLE
    print(f"\n{mod} {pres}: resuming at epoch {START_FROM}")
    for start in range(START_FROM, EPOCHS_TOTAL, CYCLE):
        span  = min(CYCLE, EPOCHS_TOTAL - start)
        model = build_joint_rnn_gru()
        safe_load_weights(model, best_path)
        print(f"Training from epoch {start} to {start + span - 1} for {mod} {pres}")
        model.fit([DEMOG[tr_idx],  ASSESS[tr_idx],  CLICKS[tr_idx]],
                  TARGET[tr_idx],
                  validation_data=([DEMOG[val_idx], ASSESS[val_idx], CLICKS[val_idx]], TARGET[val_idx]),
                  batch_size=BATCH_SAFE,
                  epochs=start + span,
                  initial_epoch=start,
                  verbose=0,
                  callbacks=[csv_logger, ckpt_save])
        print(f"Finished training span for {mod} {pres}")
        K.clear_session(); gc.collect()
    model = build_joint_rnn_gru()
    safe_load_weights(model, best_path)
    y_pred = (model.predict([DEMOG[te_idx], ASSESS[te_idx], CLICKS[te_idx]], batch_size=BATCH_SAFE) > 0.5).ravel()
    acc  = accuracy_score(TARGET[te_idx], y_pred)
    prec = precision_score(TARGET[te_idx], y_pred, zero_division=0)
    rec  = recall_score  (TARGET[te_idx], y_pred, zero_division=0)
    results.append((mod, pres, acc, prec, rec))
    print(f"{mod} {pres} → acc {acc:.3f} | prec {prec:.3f} | rec {rec:.3f}")

acc_m, prec_m, rec_m = np.mean(np.array([[a,p,r] for _,_,a,p,r in results]),0)
print("\n=== Overall ===")
print(f"Accuracy  {acc_m:.3f}")
print(f"Precision {prec_m:.3f}")
print(f"Recall    {rec_m:.3f}")


# In[ ]:


# Step 5 — Week-by-week Evaluation
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

W_START = TRAINING_CONFIG["early_week"]
W_END   = TRAINING_CONFIG["last_week"]
curves   = defaultdict(lambda: defaultdict(list))

for w in range(W_START, W_END + 1):
    assess_masked = ASSESS[te_idx].copy()
    clicks_masked = CLICKS[te_idx].copy()
    if w < W_END:
        assess_masked[:, w+1:, :] = 0
        clicks_masked[:, w+1:, :] = 0
    y_pred = (model.predict([DEMOG[te_idx], assess_masked, clicks_masked], batch_size=BATCH_SAFE, verbose=0) > 0.5).ravel()
    curves[w]["acc"].append (accuracy_score (TARGET[te_idx], y_pred))
    curves[w]["prec"].append(precision_score(TARGET[te_idx], y_pred, zero_division=0))
    curves[w]["rec"].append (recall_score   (TARGET[te_idx], y_pred, zero_division=0))

weeks   = np.arange(W_START, W_END + 1)
acc_avg = [np.mean(curves[w]["acc"])  for w in weeks]
pre_avg = [np.mean(curves[w]["prec"]) for w in weeks]
rec_avg = [np.mean(curves[w]["rec"])  for w in weeks]

import pandas as pd
metrics_df = pd.DataFrame({"week": weeks, "accuracy": acc_avg, "precision": pre_avg, "recall": rec_avg})
print("\nWeek-by-week metrics (averaged across courses)")
print(metrics_df.head())

plt.figure(figsize=(8,4))
plt.plot(weeks, acc_avg,  label="Accuracy")
plt.plot(weeks, pre_avg,  label="Precision")
plt.plot(weeks, rec_avg,  label="Recall")
plt.xlabel("Week of course (0-39)")
plt.ylabel("Metric")
plt.title("Online at-risk prediction – Joint GRU (reproduction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

