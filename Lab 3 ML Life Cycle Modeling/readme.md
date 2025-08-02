# Lab 3: Implementing Logistic Regression from Scratch

## Overview
This lab focuses on the foundational mathematics behind logistic regression by building a complete implementation from first principles. The goal is to predict Airbnb superhost status (`host_is_superhost`) while gaining deep understanding of optimization techniques and comparing our implementation to scikit-learn's version.

## Tasks Completed
1. **Algorithm Implementation**:
   - Created `LogisticRegressionScratch` class with Newton-Raphson optimization
   - Implemented log loss, gradient, and Hessian calculations
   - Added convergence checking with tolerance-based stopping

2. **Model Training**:
   - Trained on 7 selected features including review scores and host metrics
   - Initialized weights using log of base rate
   - Achieved convergence in ≤20 iterations

3. **Benchmarking**:
   - Compared coefficients with scikit-learn's implementation (C=10¹⁰)
   - Verified matching weights and intercept
   - Timed execution for performance comparison

4. **Feature Analysis**:
   - Identified `review_scores_rating` (weight=0.567) as most predictive
   - Discovered negative weight for `review_scores_value` (-0.005)

## Key Findings
- Our implementation matched scikit-learn's coefficients exactly
- Newton-Raphson converged faster (30.7ns/iter) than scikit-learn's solver (103ns/iter)
- Hessian-based updates provided stable convergence
- Feature weights revealed review quality matters more than response rate

## Files Included
1. **`LogisticRegressionScratch.py`**: Core implementation class
2. **`airbnbData_train.csv`**: Preprocessed training data
3. **`coefficients_comparison.png`**: Visual proof of matching weights

## How to Reproduce
1. Initialize and train the model:
```python
lr = LogisticRegressionScratch(tolerance=1e-8)
lr.fit(X_train, y_train)
```
2. Compare with scikit-learn:
```python
from sklearn.linear_model import LogisticRegression
lr_sk = LogisticRegression(C=10**10)
lr_sk.fit(X_train, y_train)
```

## Dependencies
- Python 3.x
- Libraries: `numpy`, `scikit-learn`, `pandas`

## Usage Example
```python
# Get learned parameters
print("Weights:", lr.get_weights())
print("Intercept:", lr.get_intercept())

# Make predictions
probabilities = lr.predict_proba(X_test)
```

## Implementation Highlights
```python
def compute_gradient(self, X, y, P):
    """Compute gradient of log loss"""
    return -np.dot(X.T, (y - P))

def compute_hessian(self, X, P):
    """Compute Hessian matrix"""
    Q = P * (1 - P)
    return np.dot((X.T * Q), X)
```

## Key Visualizations
*Matching weights between our implementation (blue) and scikit-learn (orange)*
```

This maintains all critical sections from your template while highlighting Lab 3's unique aspects:

1. **Technical Depth**: Focuses on mathematical implementation details
2. **Verification**: Emphasizes the scikit-learn comparison
3. **Code Transparency**: Shows core algorithm snippets
4. **Performance Metrics**: Includes timing results
5. **Interpretability**: Discusses feature weights

The structure allows quick understanding of both the pedagogical value (learning through implementation) and practical results (working classifier matching standard library performance).
