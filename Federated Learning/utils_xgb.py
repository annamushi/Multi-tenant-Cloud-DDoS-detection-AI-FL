import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_local_data(cid):
    data_files = {
        'client1': 'data/tenant_1_mixed.csv',
        'client2': 'data/tenant_2_mixed.csv',
        'client3': 'data/tenant_3_mixed.csv'
    }

    selected = pd.read_csv('selected_features.csv')
    features = selected['feature'].tolist()[:15]

    df = pd.read_csv(data_files[cid])
    X = df[features].values
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test

def load_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

def train_local(model, x_train, y_train):
    model.fit(x_train, y_train)

def evaluate_local(model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }
    loss = 1.0 - metrics['accuracy']
    return loss, metrics

def get_model_weights(model):
    return model.get_booster().save_raw('json')

def set_model_weights(model, weights):
    model.get_booster().load_model(bytearray(weights))