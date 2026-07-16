# Titanic Survival Prediction 

A machine learning classification project predicting passenger survival.

## What I Did
- Cleaned data: handled missing values, dropped irrelevant columns
- Encoded categorical features using get_dummies
- Trained and compared Logistic Regression vs KNN
- Tuned KNN hyperparameters using GridSearchCV + KFold cross-validation
- Built an end-to-end sklearn Pipeline (imputer → scaler → KNN)

## Results
| Model               | Accuracy | F1    | Precision | Recall |
|---------------------|----------|-------|-----------|--------|
| Logistic Regression | 79.7%    | 0.739 | 0.854     | 0.651  |
| KNN (k=11, tuned)   | 64.3%    | 0.592 | 0.597     | 0.587  |
| Pipeline (KNN)      | 80.4%    | 0.75  | 0.80      | 0.79   |

**Best model: Pipeline KNN — 80.4% accuracy**

## Libraries
pandas · numpy · scikit-learn · matplotlib

## Key Finding
Logistic Regression and the sklearn Pipeline both outperform raw KNN,
suggesting the Titanic survival problem responds better to linear 
decision boundaries than distance-based methods on this feature set.
