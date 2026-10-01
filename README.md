# Airline Passenger Satisfaction Prediction

A classification project that predicts whether an airline passenger was **satisfied** or **neutral/dissatisfied** with their flight, using their profile, trip details and in-flight service ratings. The final model, a tuned Random Forest, reaches **96.51% test accuracy** and a **ROC-AUC of 0.9948**.

I built this as part of my MSc Data Science studies. I wanted the whole workflow in one notebook: cleaning, a leak-free preprocessing pipeline, a fair model comparison, tuning, and a look at what actually drives satisfaction.

## Dataset

Airline Passenger Satisfaction survey data from Kaggle: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction (I used `train.csv`).

| | |
|---|---|
| Rows | 103,904 |
| Columns | 25 (22 used as features after dropping `Unnamed: 0`, `id` and the target) |
| Target | `satisfaction`: 58,879 neutral/dissatisfied (56.7%), 45,025 satisfied (43.3%) |
| Passenger info | Gender, Customer Type, Age |
| Trip info | Type of Travel, Class, Flight Distance, Departure Delay, Arrival Delay |
| Service ratings | 14 columns scored 0 to 5 (wifi, online boarding, seat comfort, cleanliness, baggage handling and so on) |

Only `Arrival Delay in Minutes` has missing values (310 rows, about 0.3%).

## Approach

1. **Cleaning.** Dropped `Unnamed: 0` (a row counter) and `id` (an identifier), since neither describes the passenger or the flight. Encoded the target as 1 for satisfied and 0 otherwise.
2. **EDA.** Checked the class balance and how satisfaction varies by travel class and type of travel.
3. **Preprocessing pipeline.** Median imputation and standard scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical ones. All of it lives inside a scikit-learn `Pipeline`, so the imputation values are learned from the training data only and nothing from the test set leaks in.
4. **Split.** 80/20 stratified train/test split (83,123 training rows, 20,781 test rows), `random_state=42`.
5. **Model comparison.** Logistic Regression, Linear SVM, Decision Tree and Random Forest, all on the same split with the same preprocessing. I also ran 5-fold stratified cross-validation on the training set to check the results were stable.
6. **Tuning.** `RandomizedSearchCV` on the Random Forest, using the training set only. The test set was used once, for the final evaluation.
7. **Interpretation.** Random Forest importance and permutation importance on the test set.

## EDA

![image text](images/eda.png)

Satisfaction differs a lot by group. Business class passengers are far more likely to be satisfied (roughly 69%) than Eco Plus (about 25%) or Eco (about 19%). The gap is just as large for type of travel: roughly 58% of business travellers were satisfied, against about 10% of personal travellers.

## Results

### Model comparison

| Model | Test accuracy |
|---|---|
| Random Forest (baseline) | 0.963 |
| Decision Tree | 0.946 |
| Logistic Regression | 0.877 |
| Linear SVM | 0.876 |

![Model comparison](images/model_comparison.png)

The two tree-based models are clearly ahead of the two linear ones. That suggests the relationship between the features and satisfaction is not linear. For example, the effect of a rating probably depends on the class or the reason for travel.

### Final model (tuned Random Forest)

| Metric | Value |
|---|---|
| Test accuracy | 96.51% |
| ROC-AUC | 0.9948 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Neutral/Dissatisfied | 0.959 | 0.980 | 0.970 | 11,776 |
| Satisfied | 0.973 | 0.946 | 0.959 | 9,005 |

Tuning moved the Random Forest from 0.963 to 0.965 on the test set, so most of the performance comes from the model choice and the data, not the search.

![Confusion matrix and ROC curve](images/confusion_matrix_roc.png)

Out of 20,781 test passengers, the model made 726 mistakes: 236 dissatisfied passengers predicted as satisfied, and 490 satisfied passengers predicted as dissatisfied. It misses slightly more satisfied passengers than it wrongly flags, which is why recall is a bit lower for that class.

## What drives satisfaction

I looked at feature importance in two ways, because they answer slightly different questions.

**Random Forest importance** (on encoded features, so `Type of Travel` is split into two separate columns):

![Random Forest importance](images/feature_importance_rf.png)

**Permutation importance** (on the test set, grouped by original column, measured as the drop in accuracy when a column is shuffled):

![Permutation importance](images/feature_importance_permutation.png)

The two views disagree on the order, and that is worth understanding:

- In the Random Forest chart, **Online boarding** and **Inflight wifi service** come first. `Type of Travel` looks smaller there only because its importance is split across two one-hot columns.
- In the permutation chart, where those columns are grouped, **Type of Travel** is the single most important feature, followed by **Inflight wifi service**, **Customer Type**, **Online boarding** and **Seat comfort**.
- Taken together, the trip context (why the passenger is flying and what kind of customer they are) and a few specific services (wifi, online boarding, seat comfort) matter most. Features such as gate location and age contribute very little.

## Example prediction

The pipeline accepts raw columns, so a new passenger can be scored directly. A loyal business-class passenger on a business trip, with high ratings for wifi, boarding and entertainment and no delays, is predicted as:

```
Prediction: Satisfied  (P(satisfied) = 0.77)
```

## Limitations

- The service ratings are collected from passengers **after** the flight. The model therefore explains satisfaction after the fact. It cannot predict it before departure.
- A rating of 0 means "not applicable" in this dataset, but I treated it as an ordinary number.
- The results come from one train/test split on one survey dataset. The cross-validation runs gave a sense of stability, but I have not tested the model on data from other airlines or periods.
- Feature importance shows association, not cause. Type of Travel, Class and Customer Type are related to each other, which can shift importance between them. Improving wifi will not necessarily raise satisfaction by the amount the chart suggests.

## Repository structure

```
.
├── airline_passenger_satisfaction.ipynb
├── requirements.txt
├── README.md
└── images/
```

## How to run

1. Clone the repo and install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download `train.csv` from the Kaggle link above, rename it to `Airline data set.csv` and place it next to the notebook.
3. Open `airline_passenger_satisfaction.ipynb` in Jupyter or Colab and run all cells.

In Colab, the notebook reads the file from Google Drive instead. Change the path in the loading cell if yours is different. The tuning step takes a few minutes; set `RUN_TUNING = False` in the first cell to skip it.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Possible next steps

- Try gradient boosting (XGBoost or LightGBM) and compare it with the Random Forest.
- Adjust the decision threshold to trade precision against recall, depending on what an airline cares about more.
- Add SHAP values for explanations of individual predictions.
- Save the trained pipeline with `joblib` and wrap it in a small Streamlit app.

## Author

Kaushik, MSc Data Science student at the University of Kalyani.
LinkedIn: [add your link]
