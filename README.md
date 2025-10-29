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
sms_svm_project/

├─ README.md

├─ requirements.txt

├─ LICENSE

├─ .gitignore

├─ src/

│ ├─ preprocessing.py

│ ├─ models.py

│ ├─ metrics.py

│ └─ visualization.py

├─ train.py

├─ evaluate.py

├─ bootstrap_eval.py

├─ scripts/

│ └─ run_experiments.sh

├─ data/

│ └─ (auto-downloaded) or place sms.csv with columns: label,text

├─ outputs/

│ ├─ models/

│ └─ figures/

├─ notebooks/

│ └─ Exploration.ipynb

└─ tests/

└─ test_smoke.py


### HOW TO RUN

Run first the 'environement-requirements.yml' file in order to create the environement and install the requirements
by running:


mamba env create -f environment.yml

mamba activate advanced-statistical-learning

make the scripts executable by running
chmod +x scripts/setup_env.sh
chmod +x scripts/experiments.sh
chmod +x scripts/run_all.sh

