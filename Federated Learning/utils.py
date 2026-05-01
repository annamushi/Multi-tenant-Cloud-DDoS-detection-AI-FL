import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_feature_list(path='selected_features.csv', top_n=15):
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        features = df.iloc[:, 0].tolist()
    else:
        features = df[df.columns[0]].tolist()
    return features[:top_n]


def load_local_data(cid):
    data_files = {
        'client1': 'data/tenant_1_mixed.csv',
        'client2': 'data/tenant_2_mixed.csv',
        'client3': 'data/tenant_3_mixed.csv'
    }

    df = pd.read_csv(data_files[cid])
    features = load_feature_list('selected_features.csv', top_n=15)

    X = df[features].values
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


class LogRegModel:
    def __init__(self, input_dim):
        self.model = LogisticRegression(max_iter=100, random_state=42)
        self.input_dim = input_dim
        self.is_fitted = False

    def get_weights(self):
        if not self.is_fitted:
            return [
                np.random.randn(self.input_dim, 1).astype(np.float32),
                np.zeros((1,)).astype(np.float32)
            ]
        return [
            self.model.coef_.T.astype(np.float32),
            self.model.intercept_.astype(np.float32)
        ]

    def set_weights(self, weights):
        self.model.coef_ = weights[0].T
        self.model.intercept_ = weights[1]
        self.model.classes_ = np.array([0, 1])
        self.is_fitted = True

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, x):
        return self.model.predict(x)


def load_model(input_dim):
    return LogRegModel(input_dim)


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
