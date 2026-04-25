import pickle
import streamlit as st
import numpy as np

ATTACK_COLORS = {
    'normal': '#00ff88',
    'smurf': '#ff4444',
    'neptune': '#ff6600',
    'back': '#ff0066',
    'satan': '#cc00ff',
    'ipsweep': '#ff8800',
    'portsweep': '#ff2200',
    'warezclient': '#aa00ff',
    'pod': '#ff4400',
    'teardrop': '#dd0000',
    'nmap': '#ff9900',
    'other_attack': '#ff5500',
}

MODEL_DISPLAY_NAMES = {
    'random_forest': '🌲 Random Forest',
    'decision_tree': '🌿 Decision Tree',
    'gradient_boosting': '📈 Gradient Boosting',
    'adaboost': '⚡ AdaBoost',
    'hist_gradient_boosting': '🚀 Hist Gradient Boosting',
    'knn': '🔎 K-Nearest Neighbors',
    'xgboost': '🔥 XGBoost',
}

MODEL_ACCURACIES = {
    'random_forest': 1.0000,
    'decision_tree': 1.0000,
    'gradient_boosting': 1.0000,
    'hist_gradient_boosting': 0.9895,
    'knn': 0.9983,
    'adaboost': 0.9794,
    'xgboost': 0.9990,
}


@st.cache_resource
def load_model(model_name, models_dir='models'):
    path = f'{models_dir}/{model_name}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(model, X_scaled, preprocessors):
    le_target = preprocessors['le_target']
    preds = model.predict(X_scaled)
    pred_labels = le_target.inverse_transform(preds)

    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)

    return pred_labels, proba, le_target.classes_


def is_attack(label):
    return label.lower() != 'normal'


def get_status_color(label):
    return ATTACK_COLORS.get(label.lower(), '#ff5500')


def get_attack_category(label):
    dos = ['smurf', 'neptune', 'back', 'pod', 'teardrop', 'land']
    probe = ['satan', 'ipsweep', 'portsweep', 'nmap']
    r2l = ['warezclient', 'imap', 'guess_passwd', 'warezmaster', 'ftp_write', 'multihop', 'phf', 'spy']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
    if label == 'normal':
        return 'Normal Traffic'
    elif label in dos:
        return 'DoS Attack'
    elif label in probe:
        return 'Probe/Scan'
    elif label in r2l:
        return 'R2L Attack'
    elif label in u2r:
        return 'U2R Attack'
    else:
        return 'Unknown Attack'
