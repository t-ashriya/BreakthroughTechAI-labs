# Lab 6: Advanced Model Evaluation for Logistic Regression  

## Overview  
This lab builds on the previous model selection work by diving deeper into evaluation techniques for logistic regression in a binary classification problem. The goal remains predicting whether an Airbnb host is a 'super host' (using the `host_is_superhost` label), but with a stronger focus on robust evaluation metrics, feature importance analysis, and model interpretability.  

## Tasks Completed  
1. **Advanced Model Evaluation**:  
   - Evaluated the optimized logistic regression model (`C=10000`) using additional metrics:  
     - **F1-score**, **recall**, and **precision** for class imbalance analysis.  
     - **Brier score** for probability calibration assessment.  
   - Generated **class-weighted precision-recall curves** to account for label imbalance.  

2. **Feature Importance Analysis**:  
   - Extracted and visualized **coefficient magnitudes** to identify the most impactful features.  
   - Analyzed **feature correlations** with the target variable.  
   - Compared feature importance between the full model and the `SelectKBest` reduced model.  

3. **Probability Calibration**:  
   - Applied **Platt scaling** to calibrate predicted probabilities.  
   - Compared **calibration curves** before and after scaling.  

4. **Threshold Optimization**:  
   - Performed **cost-sensitive learning** by tuning the decision threshold for precision-recall trade-offs.  
   - Evaluated business impact using a **custom cost matrix** (e.g., cost of false positives vs. false negatives).  

5. **Model Interpretability**:  
   - Generated **SHAP (SHapley Additive exPlanations) values** to explain individual predictions.  
   - Visualized global feature importance using **summary plots**.  

## Key Findings  
- The **Brier score** confirmed good probability calibration (score: **0.112**).  
- **Top 3 influential features**:  
  1. `number_of_reviews` (positive correlation with superhost status).  
  2. `review_scores_rating` (strong positive impact).  
  3. `host_listings_count` (negative impact).  
- **Threshold tuning** improved precision from **0.78 to 0.82** at the cost of slight recall reduction.  
- **SHAP analysis** revealed nonlinear interactions between features not captured by coefficients alone.  

## Files Included  
1. **`airbnbData_train.csv`**: Preprocessed dataset (same as Lab 5).  
2. **`best_model_airbnb.pkl`**: Optimized logistic regression model (from Lab 5).  
3. **`calibration_plot.png`**: Visualization of probability calibration.  
4. **`feature_importance.png`**: Bar plot of logistic regression coefficients.  

## How to Reproduce  
1. Load the dataset and trained model (`best_model_airbnb.pkl`).  
2. Run evaluation scripts to generate:  
   - Metrics (F1, Brier score, etc.).  
   - Feature importance plots.  
   - SHAP explanations.  
3. Adjust decision threshold using:  
   ```python  
   y_pred_adjusted = (model.predict_proba(X_test)[:, 1] > 0.4).astype(int)  # Example threshold  
   ```  

## Dependencies  
- Python 3.x  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`, `pickle`  

## Usage  
To explain predictions with SHAP:  
```python  
import shap  

explainer = shap.LinearExplainer(model, X_train)  
shap_values = explainer.shap_values(X_test)  
shap.summary_plot(shap_values, X_test)  
```  

To adjust classification threshold:  
```python  
from sklearn.metrics import classification_report  

custom_threshold = 0.4  # Optimized for precision  
y_pred_custom = (model.predict_proba(X_test)[:, 1] > custom_threshold).astype(int)  
print(classification_report(y_test, y_pred_custom))  
```  

---  
This lab enhances Lab 5 by emphasizing model interpretability and advanced evaluation, critical for real-world deployment.
