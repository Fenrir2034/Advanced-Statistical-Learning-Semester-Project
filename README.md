# Advanced-Statistical-Learning-Semester-Project
Final project for the advanced statistical learning class of 2025-2026.

# SMS Spam Classification using Machine Learning - PYTHON -

This repository contains **machine learning pipeline** for detecting spam text messages (SMS).  
It demonstrates **text preprocessing**, **TF-IDF vectorization**, **model selection**, and **resampling-based evaluation** (cross-validation and ROC/PR analysis) using Python.

---

## Project Overview

Spam filtering is a classic **binary classification** problem in Natural Language Processing (NLP).  
Here, we build and compare several models to classify messages as **ham (legitimate)** or **spam (unwanted)**:

- **Logistic Regression** (with L1/L2 regularization)  
- **Linear SVM** (with and without probability calibration)  
- **Random Forest** (as a non-linear baseline)

Each model is tuned with **5-fold cross-validation**, then evaluated on a held-out test set.  
Performance is visualized using **confusion matrices**, **ROC/PR curves**, and summary tables (AUC, AP).

---

## the Dataset

The project uses the [**SMS Spam Collection Dataset**](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  
from the UCI Machine Learning Repository.

- **Size:** 5,574 messages  
- **Classes:** `ham` (≈86%) and `spam` (≈14%)  
- **Format:** tab-separated file with columns: `label`, `text`

If the dataset is not present locally, it will be **automatically downloaded** by the script.

---

### Structure
sms_spam_project/
│
├── sms_spam_project.py        # main training & evaluation script
├── requirements.txt           # dependencies
├── outputs/                   # auto-created folder with results (plots, metrics)
│   ├── logit_confusion_matrix.png
│   ├── linear_svm_roc.png
│   ├── pr_all.png
│   └── metrics_summary.csv
│
├── README.md                  # project documentation
└── data/                      # optional folder for local CSV input (label,text) if not downloaded in code

