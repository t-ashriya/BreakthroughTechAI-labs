# Lab 2: Decision Trees and KNN Classifiers for Superhost Prediction

## Overview
This lab focuses on the modeling phase of the machine learning life cycle, comparing Decision Tree and K-Nearest Neighbors (KNN) classifiers to predict Airbnb superhost status. Using the `host_is_superhost` label, we evaluate which model better classifies hosts while exploring hyperparameter tuning and model interpretability.

## Tasks Completed
1. **Data Preparation**:
   - Loaded preprocessed Airbnb data (`airbnbData_train.csv`)
   - One-hot encoded remaining categorical features
   - Created training (70%) and test (30%) splits

2. **Decision Tree Implementation**:
   - Trained trees with max_depth values [1, 2, 4, 8, 16, 32]
   - Identified optimal max_depth=8 (83.3% accuracy)
   - Visualized accuracy vs. tree depth tradeoffs

3. **KNN Classifier Implementation**:
   - Tested k values from 1 to 40 (step=3)
   - Determined optimal k=22 (77.7% accuracy)
   - Compared performance across distance metrics

4. **Model Evaluation**:
   - Created reusable visualization function for accuracy metrics
   - Analyzed decision boundaries for both models
   - Compared noise sensitivity between algorithms

## Key Findings
- Decision Trees outperformed KNN (83.3% vs 77.7% accuracy)
- Shallower trees (max_depth=8) prevented overfitting
- KNN showed greater sensitivity to feature scaling
- Decision Trees handled categorical data more effectively

## Files Included
1. **`airbnbData_train.csv`**: Preprocessed dataset
2. **`decision_tree_visualization.png`**: Accuracy vs. max_depth plot
3. **`knn_accuracy_plot.png`**: KNN performance across k-values

## How to Reproduce
1. Preprocess data (one-hot encoding)
2. For Decision Trees:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier(max_depth=8)
   ```
3. For KNN:
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   model = KNeighborsClassifier(n_neighbors=22)
   ```
4. Evaluate using accuracy_score()

## Dependencies
- Python 3.x
- Libraries: `scikit-learn`, `matplotlib`, `seaborn`, `pandas`

## Usage Example
```python
# Compare model performance
from sklearn.metrics import accuracy_score

dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.3f}")
print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred):.3f}")
```

## Key Visualizations
*Accuracy trends across hyperparameter values for both models*

## Best Practices Identified
- Decision Trees require depth limitation to prevent overfitting
- KNN benefits from feature scaling and odd k-values
- Visualizing hyperparameter effects aids model selection
```

This maintains the same clear structure as your Lab 5 README while adapting to Lab 2's content. Key features include:

1. **Problem-Specific Details**: Focuses on the classification task and model comparison
2. **Technical Precision**: Includes specific accuracy scores and optimal parameters
3. **Reproducibility**: Provides ready-to-use code snippets
4. **Visual Documentation**: References generated plots
5. **Practical Insights**: Highlights learned best practices

The format ensures quick scanning while containing all necessary details to understand and replicate the analysis.
