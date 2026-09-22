# MSc in AI Capstone #3: Machine Learning Foundations Project

**Repository:** https://github.com/PHIacademy/applied-machine-learning.git

## Project Description

This project builds a complete, reproducible supervised machine learning workflow on the UCI Adult / Census Income dataset. It loads and cleans 45,222 census records, prepares the data for modeling (skew correction, scaling, one-hot encoding), trains and compares three classifiers (Decision Tree, Support Vector Classifier, Logistic Regression), tunes the selected model via grid search, and evaluates feature importance to test whether a reduced feature set retains the model's predictive power. The use case frames the model as a tool for **phiAI**, a nonprofit, to prioritize donor outreach for its STEM scholarship programs by identifying individuals likely to earn more than $50,000 annually.

## File Structure

```
.
├── README.md
├── Machine_Learning_Analysis_Report.pdf
├── modeling.ipynb
├── visuals.py
├── census.csv
└── requirements.txt
```

- `modeling.ipynb` — a Jupyter Notebook containing the full workflow: data ingestion, data exploration and quality checks (missing values, duplicates, range checks), preprocessing (log transformation, min-max scaling, one-hot encoding), training and comparison of three supervised learning models, hyperparameter tuning via grid search, feature importance analysis and feature selection, and a final notebook summary with a limitations/bias discussion.
- `visuals.py` — supplementary plotting code used by the notebook to visualize skewed feature distributions, model comparison metrics, and feature importance.
- `requirements.txt` — the Python dependencies needed to run the notebook.
- `Machine_Learning_Analysis_Report.pdf` — a written APA-formatted report with academic citations, covering the overview, dataset description, modeling approach, results, a non-technical interpretation, and limitations/bias discussion (see separate report).

## Dataset

**UCI Adult / Census Income Dataset** (1994 U.S. Census Bureau demographic and employment records)
Source: [UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult)
File used: `census.csv`

## How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Get the dataset

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) and place it as `census.csv` in the same folder as `modeling.ipynb`.

### 3. Run the notebook

```
jupyter notebook modeling.ipynb
```

Run all cells from top to bottom. The notebook should execute without errors and reproduce all preprocessing steps, model comparison tables, tuning results, and feature importance visualizations.
