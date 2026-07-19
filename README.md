# Titanic Survival Prediction 

Predicting passenger survival using both classical ML and a PyTorch neural network.

## Notebooks
| Notebook | Approach |
|---|---|
| `titanic-survival-ml.ipynb` | Classical ML — 4 models compared |
| `titanic-survival-nn.ipynb` | Neural Network — PyTorch |

## Results
| Model                 | Accuracy |
|-----------------------|----------|
| Neural Network        | 82.0%    |
| XGBoost (tuned)       | 80.9%    |
| Pipeline KNN          | 78.7%    |
| Logistic Regression   | 78.1%    |
| Random Forest (tuned) | 74.7%    |
| KNN (tuned)           | 70.0%    |

**Best model: Neural Network — 82.0%**

## Key Findings
- Neural Network outperformed all classical ML models
- XGBoost is the best classical model — complex models benefit from more data
- Feature engineering (Deck, TicketCount) improved results over raw features
- Fare and Age were the strongest predictors per XGBoost feature importance

## Stack
**ML:** pandas · scikit-learn · XGBoost · matplotlib  
**NN:** PyTorch · scikit-learn · matplotlib
