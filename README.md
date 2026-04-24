# 🛡️ Network Intrusion Detection System (NIDS)

A high-performance **Network Intrusion Detection System (NIDS)** built using classical machine learning and ensemble techniques to classify network traffic as **normal** or **malicious (attack types)**.

This project benchmarks multiple models on the **KDD Cup 1999 dataset**, delivering near state-of-the-art performance with detailed evaluation metrics and visual diagnostics.

---

## 🚀 Project Highlights

- 🔍 Multi-class classification of network traffic
- ⚡ High accuracy (up to **99.76%**)
- 🤖 Comparison of multiple ML algorithms
- 📊 Advanced evaluation metrics + confusion matrices
- 🧪 Scalable and reproducible pipeline

---

## 📂 Dataset

- **KDD Cup 1999 Dataset (10% subset)**
- Widely used benchmark in intrusion detection research

### Files Used:
- `KDDCup Data 10 Percent.csv` → Main dataset  
- `kddcup.txt` → Feature/column names  
- `training_attack_types.txt` → Attack label mappings  

---

## 🧠 Models Implemented

| Model                     | Description |
|--------------------------|------------|
| Decision Tree            | Baseline interpretable model |
| Random Forest            | Ensemble of decision trees |
| K-Nearest Neighbors      | Distance-based classifier |
| AdaBoost                 | Boosting-based weak learners |
| Gradient Boosting        | Sequential error correction |
| HistGradientBoosting     | Optimized gradient boosting |
| XGBoost                  | High-performance boosting |

---

## 📊 Performance Metrics

| Model                     | Accuracy |
|--------------------------|----------|
| 🥇 Random Forest          | **99.76%** |
| 🥈 Decision Tree          | 99.59% |
| 🥉 KNN                    | 99.41% |
| XGBoost                  | 99.39% |
| AdaBoost                 | 94.54% |
| HistGradientBoosting     | 93.10% |
| Gradient Boosting        | 92.86% |

---

## 📈 Evaluation Strategy

Each model is evaluated using:

- ✅ Accuracy  
- 🎯 Precision  
- 🔁 Recall  
- ⚖️ F1-Score  
- 🔲 Confusion Matrix Visualization  

---

## 🛠️ Tech Stack

### Core Libraries
- `pandas` – Data processing  
- `numpy` – Numerical computations  

### Visualization
- `matplotlib` – Plotting  
- `seaborn` – Statistical visualization  

### Machine Learning
- `scikit-learn`
  - DecisionTreeClassifier
  - RandomForestClassifier
  - AdaBoostClassifier
  - GradientBoostingClassifier
  - HistGradientBoostingClassifier
  - KNeighborsClassifier
  - StandardScaler
  - LabelEncoder
  - GridSearchCV  

- `xgboost` – Advanced boosting algorithm  

---

## ⚙️ How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/NIDS.git
cd NIDS
