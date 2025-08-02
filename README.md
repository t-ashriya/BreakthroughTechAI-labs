# BTTAI-Machine Learning Labs Portfolio

## Overview
This repository contains implementations of key machine learning concepts across the ML lifecycle, from data preparation to advanced modeling techniques. All labs use the Airbnb listings dataset (except Lab 7 which uses MNIST) to solve real-world prediction problems.

## Lab 1: Data Understanding and Preparation
**Key Concepts:**
- Data cleaning (missing value imputation, outlier handling via winsorization)
- Feature engineering (one-hot encoding)
- Exploratory data analysis (correlation analysis, visualization)

**Techniques Applied:**
- Winsorized price outliers (`scipy.stats.mstats.winsorize`)
- Mean imputation for missing values
- One-hot encoding categorical features (`pd.get_dummies`)
- Correlation analysis and pairplot visualization

**Key Findings:**
- Identified `accommodates` and `bedrooms` as most correlated with price (0.50 and 0.42 correlation)
- Text features like listing descriptions contain valuable unstructured data for NLP processing

---

## Lab 2: Decision Trees and KNN Classifiers
**Key Concepts:**
- Hyperparameter tuning for decision trees (max_depth)
- K-nearest neighbors implementation
- Model evaluation and comparison

**Techniques Applied:**
- Grid search for optimal decision tree depth (best: max_depth=8)
- KNN with varying k values (optimal k=22)
- Accuracy metric visualization

**Key Findings:**
- Decision Tree (max_depth=8) achieved 83.3% accuracy
- KNN (k=22) achieved 77.7% accuracy
- Decision Trees handled noisy data better than KNN

---

## Lab 3: Logistic Regression from Scratch
**Key Concepts:**
- Implementation of logistic regression core components:
  - Log loss, gradient, and Hessian calculations
  - Newton-Raphson optimization
- Comparison with scikit-learn's implementation

**Techniques Applied:**
- Custom `LogisticRegressionScratch` class
- Weight updates via Hessian matrix
- Benchmarking against `sklearn.linear_model.LogisticRegression`

**Key Findings:**
- Matching coefficients with scikit-learn (validated implementation)
- Custom implementation was faster (30.7ns vs 103ns per loop)

---

## Lab 4: Model Selection for Logistic Regression
**Key Concepts:**
- Hyperparameter tuning (regularization strength C)
- Model evaluation metrics (AUC-ROC, precision-recall)
- Feature selection (SelectKBest)

**Techniques Applied:**
- Grid search for optimal C value (best: C=10000)
- Precision-recall and ROC curve visualization
- Top 5 feature selection using ANOVA F-value

**Key Findings:**
- Optimal model achieved AUC of 0.823 (vs 0.821 baseline)
- Best features: host_response_rate, number_of_reviews metrics
- Full feature set (49 features) performed best

---

## Lab 5: Comparing Regression Models
**Key Concepts:**
- Regression model comparison (Linear, Decision Tree, Ensemble methods)
- Evaluation metrics (RMSE, R²)
- Ensemble methods (Stacking, Random Forest, GBDT)

**Techniques Applied:**
- Stacking (Linear Regression + Decision Tree)
- Random Forest with max_depth=32
- Gradient Boosted Trees with n_estimators=300

**Key Findings:**
- Random Forest performed best (RMSE: 0.631, R²: 0.622)
- Ensemble methods outperformed individual models
- Stacking showed modest improvement over base models

---

## Lab 6: Implementing a CNN with Keras
**Key Concepts:**
- CNN architecture design
- Image data preprocessing
- Neural network training and evaluation

**Techniques Applied:**
- 4-layer CNN with increasing filters (16→128)
- Global Average Pooling
- SGD optimizer with learning rate 0.1

**Key Findings:**
- Achieved 91.4% test accuracy on MNIST
- Training time: 68.62s for 1 epoch
- Effective digit recognition as shown in sample predictions

---

## Lab 7: Model Persistence and Deployment
**Key Concepts:**
- Model serialization
- Production-ready model packaging
- Prediction serving

**Techniques Applied:**
- Pickle serialization (`best_model_airbnb.pkl`)
- Model reloading and validation
- GitHub deployment workflow

**Key Findings:**
- Successful model persistence with consistent predictions
- Demonstrated end-to-end deployment pipeline

---

## How to Run
1. Clone repository
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `seaborn`, `matplotlib`
3. Run Jupyter notebooks in order (Lab 1 → Lab 7)

## Dataset Information
- **Primary Dataset**: Airbnb NYC listings (preprocessed)
  - Classification: Superhost prediction (Labs 2-5)
  - Regression: Price prediction (Lab 6)
- **Secondary Dataset**: MNIST handwritten digits (Lab 7)

## Key Takeaways
- Demonstrated full ML lifecycle from raw data to deployed models
- Implemented diverse algorithms from logistic regression to CNNs
- Emphasized model evaluation and comparison
- Highlighted importance of hyperparameter tuning
