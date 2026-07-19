# Titanic Survival Prediction 

Predicting passenger survival using both classical ML and a PyTorch neural network.

## Notebooks
| Notebook | Approach |
|---|---|
| `titanic-survival-ml.ipynb` | Classical ML — 5 models compared |
| `titanic-survival-nn.ipynb` | Neural Network — PyTorch |

## Results
| Model                 | Accuracy |
|-----------------------|----------|
| Neural Network        | 82.0%    |
| XGBoost (tuned)       | 80.9%    |
| Pipeline KNN          | 79.8%    |
| Logistic Regression   | 79.2%    |
| Random Forest (tuned) | 77.5%    |
| KNN (tuned)           | 70.8%    |

**Best model: Neural Network — 82.0%**

## Key Findings
- Neural Network outperformed all classical ML models
- XGBoost is the best classical model — complex models benefit from more data
- Feature engineering (Deck, TicketCount) improved results over raw features
- Fare and Age were the strongest predictors per XGBoost feature importance

## Stack
**ML:** pandas · scikit-learn · XGBoost · matplotlib  
**NN:** PyTorch · scikit-learn · matplotlib
