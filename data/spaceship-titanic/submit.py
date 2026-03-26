"""
Spaceship Titanic - Best model submission script
Reproduces the F1=0.8139 approach: XGBoost with cabin parsing, target encoding, spending features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from pathlib import Path

DATA_DIR = Path(__file__).parent

def engineer_features(df):
    """Feature engineering pipeline."""
    # Cabin parsing -> Deck, CabinNum, Side
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce')
    df['Side'] = df['Cabin'].str.split('/').str[2]
    
    # PassengerId -> Group
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    
    # Spending features
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['SpendPerAge'] = df['TotalSpend'] / (df['Age'] + 1)
    
    # CryoSleep interactions (cryo passengers have zero spend)
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    
    # Age binning
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 80], labels=[0,1,2,3,4]).astype(float)
    
    return df

def target_encode(train_df, val_df, col, target, smoothing=10):
    """Target encoding with smoothing."""
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    train_encoded = train_df[col].map(smooth).fillna(global_mean)
    val_encoded = val_df[col].map(smooth).fillna(global_mean)
    return train_encoded, val_encoded

def prepare_features(df, is_train=True):
    """Prepare feature matrix."""
    df = engineer_features(df.copy())
    
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'CabinNum', 'TotalSpend', 'HasSpend', 'SpendPerAge', 'GroupSize', 'AgeBin',
                'CryoSleep', 'VIP']
    cat_cols_te = ['Deck', 'Side']  # target encoded
    cat_cols_ohe = ['HomePlanet', 'Destination']  # one-hot encoded
    
    return df, num_cols, cat_cols_te, cat_cols_ohe

# Load data
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

train['Transported'] = train['Transported'].astype(int)

# CV to verify score
train_prep, num_cols, cat_cols_te, cat_cols_ohe = prepare_features(train)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = train_prep['Transported']

f1_scores = []
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_prep, y)):
    tr = train_prep.iloc[train_idx].copy()
    va = train_prep.iloc[val_idx].copy()
    
    # Target encode
    for col in cat_cols_te:
        tr[f'{col}_te'], va[f'{col}_te'] = target_encode(tr, va, col, 'Transported')
    
    # One-hot encode
    tr_ohe = pd.get_dummies(tr[cat_cols_ohe], drop_first=True)
    va_ohe = pd.get_dummies(va[cat_cols_ohe], drop_first=True)
    va_ohe = va_ohe.reindex(columns=tr_ohe.columns, fill_value=0)
    
    # Build feature matrices
    te_cols = [f'{c}_te' for c in cat_cols_te]
    X_tr = pd.concat([tr[num_cols].fillna(0), tr[te_cols], tr_ohe], axis=1)
    X_va = pd.concat([va[num_cols].fillna(0), va[te_cols], va_ohe], axis=1)
    
    # Scale
    scaler = MinMaxScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_va = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns, index=X_va.index)
    
    y_tr = tr['Transported']
    y_va = va['Transported']
    
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    
    f1 = f1_score(y_va, preds)
    acc = accuracy_score(y_va, preds)
    f1_scores.append(f1)
    acc_scores.append(acc)
    print(f'Fold {fold+1}: F1={f1:.4f}, Acc={acc:.4f}')

print(f'\nCV Mean F1={np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})')
print(f'CV Mean Acc={np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})')

# Train on full data and predict test
print('\nTraining final model on full data...')
train_full = engineer_features(train.copy())
test_full = engineer_features(test.copy())

# Target encode on full train
global_mean = train_full['Transported'].mean()
for col in cat_cols_te:
    agg = train_full.groupby(col)['Transported'].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
    train_full[f'{col}_te'] = train_full[col].map(smooth).fillna(global_mean)
    test_full[f'{col}_te'] = test_full[col].map(smooth).fillna(global_mean)

# One-hot
tr_ohe = pd.get_dummies(train_full[cat_cols_ohe], drop_first=True)
te_ohe = pd.get_dummies(test_full[cat_cols_ohe], drop_first=True)
te_ohe = te_ohe.reindex(columns=tr_ohe.columns, fill_value=0)

te_cols = [f'{c}_te' for c in cat_cols_te]
X_full = pd.concat([train_full[num_cols].fillna(0), train_full[te_cols], tr_ohe], axis=1)
X_test = pd.concat([test_full[num_cols].fillna(0), test_full[te_cols], te_ohe], axis=1)

scaler = MinMaxScaler()
X_full = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

final_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, verbosity=0
)
final_model.fit(X_full, train_full['Transported'])
test_preds = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_preds.astype(bool)
})
submission.to_csv(DATA_DIR / 'submission.csv', index=False)
print(f'\nSubmission saved: {DATA_DIR / "submission.csv"}')
print(f'Predictions: {sum(test_preds)} transported, {len(test_preds) - sum(test_preds)} not')
print(submission.head())
