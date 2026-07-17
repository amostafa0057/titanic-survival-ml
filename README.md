# Titanic Survival Prediction 🚢

Predicting whether a passenger survived the Titanic using machine learning.

## Approach
Trained and compared 4 classifiers on the Titanic dataset after cleaning 
and encoding. Used GridSearchCV with KFold cross-validation for tuning, 
and built a full sklearn Pipeline for the best KNN configuration.

## Results
| Model                 | Accuracy | F1    | Precision | Recall |
|-----------------------|----------|-------|-----------|--------|
| Pipeline KNN          | 80.4%    | 0.750 | 0.800     | 0.790  |
| Logistic Regression   | 79.7%    | 0.739 | 0.854     | 0.651  |
| XGBoost (tuned)       | 79.0%    | 0.746 | 0.800     | 0.698  |
| Random Forest (tuned) | 77.6%    | 0.738 | 0.763     | 0.714  |
| KNN (tuned, k=11)     | 64.3%    | 0.592 | 0.597     | 0.587  |

## Key Finding
Fare and Age were the strongest predictors — Fare acts as a proxy for 
wealth and class, while Age captures the "children first" evacuation pattern.
Despite being more complex, ensemble models didn't outperform Logistic 
Regression — simpler models can win on small, clean, linear datasets.

## Stack
Python · pandas · scikit-learn · XGBoost · matplotlib

## Skills Demonstrated
- Data cleaning & feature engineering
- Supervised learning (classification)
- Hyperparameter tuning (GridSearchCV + KFold)
- Ensemble methods (Random Forest, XGBoost)
- sklearn Pipelines
- Model evaluation (accuracy, F1, precision, recall, confusion matrix)
