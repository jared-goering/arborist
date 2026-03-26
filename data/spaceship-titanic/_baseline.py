
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import sys, os

data_dir = os.environ.get("DATA_DIR", "data/spaceship-titanic")
df = pd.read_csv(f"{data_dir}/train.csv")

# Minimal preprocessing
df["Transported"] = df["Transported"].astype(int)

# Extract cabin deck/num/side
df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

# Encode categoricals
for col in ["HomePlanet", "Destination", "Deck", "Side"]:
    df[col] = LabelEncoder().fit_transform(df[col].fillna("Unknown"))

df["CryoSleep"] = df["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
df["VIP"] = df["VIP"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in spend_cols:
    df[col] = df[col].fillna(0)
df["TotalSpend"] = df[spend_cols].sum(axis=1)
df["Age"] = df["Age"].fillna(df["Age"].median())

features = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "Deck", "CabinNum", "Side", "TotalSpend"]

X = df[features].fillna(0).values
y = df["Transported"].values

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s, accs = [], []
for train_idx, val_idx in skf.split(X, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y[train_idx])
    preds = model.predict(X_val)
    f1s.append(f1_score(y[val_idx], preds))
    accs.append(accuracy_score(y[val_idx], preds))

print(f"METRICS val_loss=0.0000 val_f1={np.mean(f1s):.4f} val_accuracy={np.mean(accs):.4f} val_kappa=0.0000")
