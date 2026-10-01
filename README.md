# Airline Passenger Satisfaction Prediction

An end-to-end machine learning project that predicts whether an airline passenger was **satisfied** or **neutral/dissatisfied** after a flight. The analysis compares four classifiers, tunes a Random Forest, and examines which passenger, trip, and service features are most associated with satisfaction.

> **Best result:** tuned Random Forest with **96.51% test accuracy** and **0.9948 ROC-AUC** on a stratified hold-out set.

The project is presented as a reproducible Jupyter notebook, with the dataset and generated charts included in this repository.

## Results at a glance

| Model | Test accuracy |
|---|---:|
| Tuned Random Forest | **96.51%** |
| Random Forest baseline | 96.3% |
| Decision Tree | 94.6% |
| Logistic Regression | 87.7% |
| Linear SVM | 87.6% |

The tuned model's ROC-AUC is **0.9948**. On the 20,781-row test set, it correctly classified 20,055 passengers. Per-class precision, recall, and F1 are shown below.

| Passenger group | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Neutral/dissatisfied | 0.959 | 0.980 | 0.970 | 11,776 |
| Satisfied | 0.973 | 0.946 | 0.959 | 9,005 |

These are results from one hold-out split of this dataset, not a guarantee of performance on future or other-airline data.

## Visual results

### Exploratory analysis

![Exploratory analysis of the passenger satisfaction data](images/eda.png)

### Model comparison

![Test accuracy comparison across four classifier families](images/model_comparison.png)

### Final model evaluation

![Confusion matrix and ROC curve for the tuned Random Forest](images/confusion_matrix_roc.png)

### Feature importance

![Random Forest impurity-based feature importance](images/feature_importance_rf.png)

![Permutation importance grouped by original feature](images/feature_importance_permutation.png)

The feature-importance views answer different questions. Random Forest impurity importance ranks individual encoded columns, while permutation importance evaluates each original input column by the drop in test accuracy after shuffling it. In the latter, **Type of Travel**, **Inflight wifi service**, **Customer Type**, **Online boarding**, and **Seat comfort** are among the strongest predictors. These are associations, not causal effects.

## Dataset

The dataset is the Airline Passenger Satisfaction survey dataset published on [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). The CSV used by the notebook is included as `Airline data set.csv`.

| Detail | Value |
|---|---|
| Rows | 103,904 |
| Columns | 25 |
| Target | `satisfaction` (`satisfied` vs. `neutral or dissatisfied`) |
| Class balance | 45,025 satisfied (43.3%); 58,879 neutral/dissatisfied (56.7%) |
| Missing values | 310 in `Arrival Delay in Minutes` (about 0.3%) |
| Feature groups | Passenger profile, trip details, and 14 in-flight service ratings |

The `Unnamed: 0` row index and `id` identifier are removed before modeling. The target is encoded as 1 for satisfied and 0 otherwise.

## Method

1. Explore class balance and satisfaction by travel class and travel type.
2. Split the data into stratified 80% training and 20% test sets (`random_state=42`).
3. Build preprocessing into a scikit-learn pipeline: median imputation and scaling for numeric columns, and most-frequent imputation plus one-hot encoding for categorical columns.
4. Compare Logistic Regression, Linear SVM, Decision Tree, and Random Forest on the same split; use stratified cross-validation on the training set.
5. Tune the Random Forest with `RandomizedSearchCV` using training data only. Reserve the test set for final evaluation.
6. Interpret the selected model with built-in feature importance and test-set permutation importance.

Because preprocessing is fitted inside the pipeline, imputation and encoding are learned from training data rather than the held-out test set.

## Run the notebook

The CSV is included, so no separate dataset download is needed.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies and open the notebook:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux: use `source .venv/bin/activate` instead
python -m pip install -r requirements.txt
jupyter notebook Airline_Passenger_Satisfaction_Revised.ipynb
```

Run the cells from top to bottom. The hyperparameter search can take a few minutes; set `RUN_TUNING = False` in the setup cell to skip it. In Colab, upload the CSV or update `DATA_PATH` in the data-loading cell to point to its location.

## Repository contents

```text
.
|-- Airline data set.csv
|-- Airline_Passenger_Satisfaction_Revised.ipynb
|-- README.md
|-- requirements.txt
`-- images/
   |-- confusion_matrix_roc.png
   |-- eda.png
   |-- feature_importance_permutation.png
   |-- feature_importance_rf.png
   `-- model_comparison.png
```

## Limitations

- Service ratings are collected after the flight, so this analysis explains reported post-flight satisfaction; it is not a pre-flight prediction system.
- A rating of 0 represents "not applicable" in the source data and is treated as a numeric value in this analysis.
- Evaluation uses one survey dataset and one hold-out split. Performance on other airlines, populations, or time periods has not been established.
- Feature importance indicates predictive association, not causation. Related inputs such as travel type, class, and customer type can share or shift importance.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter.

## Author

Kaushik | MSc Data Science project
