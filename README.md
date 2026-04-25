# 🛡️ NIDS — Network Intrusion Detection System

A production-grade Streamlit app for detecting network intrusions using machine learning on the KDD Cup 1999 dataset.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
nids-streamlit-app/
├── app.py                  # Main Streamlit application
├── models/
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── adaboost.pkl
│   ├── hist_gradient_boosting.pkl
│   ├── knn.pkl
│   ├── xgboost.pkl
│   └── preprocessors.pkl   # LabelEncoders + StandardScaler
├── utils/
│   ├── preprocessing.py    # Feature processing pipeline
│   └── helpers.py          # Model loading + prediction helpers
├── data/
│   └── sample.csv          # Sample KDD Cup data
└── requirements.txt
```

## 🎯 Features

- **7 ML Models**: Random Forest, Decision Tree, Gradient Boosting, AdaBoost, Hist GB, KNN, XGBoost
- **2 Input Modes**: Upload CSV batch analysis OR manual form entry
- **Real-time Predictions**: Class label + confidence scores
- **Visual Analytics**: Pie charts, bar charts, feature importance
- **Dark Cybersecurity Theme**: Neon-accented responsive dashboard
- **CSV Download**: Export all predictions

## 🔐 Attack Types Detected

normal, smurf, neptune, back, satan, ipsweep, portsweep, warezclient, pod, teardrop, nmap

## 📊 Dataset

KDD Cup 1999 (10% subset) — 41 features, 24,861 records
## App link 
https://eljapkswaklnaxgzedbngm.streamlit.app/
