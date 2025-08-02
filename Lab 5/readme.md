# Lab 5: Model Selection for Logistic Regression

## Overview
This lab focuses on the evaluation phase of the machine learning life cycle, specifically model selection for logistic regression to solve a binary classification problem. The goal is to predict whether an Airbnb host is a 'super host' using the `host_is_superhost` column as the label. The lab covers data preparation, model training, hyperparameter tuning, evaluation, and deployment.

## Tasks Completed
1. **Data Preparation**:
   - Loaded the preprocessed Airbnb listings dataset (`airbnbData_train.csv`).
   - Defined the label (`host_is_superhost`) and features (all remaining columns).
   - Split the data into training (70%) and test (30%) sets.

2. **Model Training and Evaluation**:
   - Trained a logistic regression model with default hyperparameter `C=1.0`.
   - Evaluated the model using accuracy, confusion matrix, precision-recall curves, and ROC-AUC metrics.

3. **Hyperparameter Tuning**:
   - Performed a grid search with 5-fold cross-validation to find the optimal value for hyperparameter `C`.
   - Identified the best `C` value (`C=10000`) and retrained the model.

4. **Model Comparison**:
   - Compared the performance of the default and optimized models using precision-recall curves and ROC-AUC scores.
   - The optimized model achieved a slightly higher AUC (0.823) compared to the default model (0.821).

5. **Feature Selection**:
   - Used `SelectKBest` to identify the top features for the model.
   - Tested different numbers of features (5, 10, 20, 30, 40, 49) and observed that using all 49 features yielded the best AUC score (0.823).

6. **Model Persistence**:
   - Saved the optimized model (`model_best`) to a `.pkl` file using `pickle`.
   - Loaded the model back and verified its predictions on the test set.

## Key Findings
- The optimal hyperparameter `C` for logistic regression was found to be `10000`.
- Including all 49 features resulted in the highest model performance (AUC = 0.823).
- The precision-recall and ROC curves demonstrated the improved performance of the optimized model over the default one.

## Files Included
1. **`airbnbData_train.csv`**: Preprocessed dataset used for training and testing.
2. **`best_model_airbnb.pkl`**: Pickle file containing the trained logistic regression model with the optimal hyperparameter.

## How to Reproduce
1. Load the dataset and split it into training and test sets.
2. Train the logistic regression model with default and optimized hyperparameters.
3. Evaluate the models using the provided metrics and visualizations.
4. Use the saved `.pkl` file to load the model for future predictions.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pickle`

## Usage
To use the saved model:
```python
import pickle

# Load the model
with open("best_model_airbnb.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make predictions
predictions = loaded_model.predict(X_test)
