# Lab 1: Data Understanding and Preparation

## Overview
This lab focuses on the initial phases of the machine learning life cycle, preparing raw Airbnb listing data for modeling through comprehensive data cleaning and feature engineering. The goal is to transform the dataset into a structured format suitable for predicting listing prices, while identifying key relationships between features and the target variable.

## Tasks Completed
1. **Data Inspection**:
   - Loaded the raw Airbnb NYC listings dataset (`airbnbData.csv`)
   - Identified 74 initial features with mixed data types
   - Detected missing values across numerical and categorical columns

2. **Data Cleaning**:
   - Winsorized price outliers at 1% thresholds to create `label_price`
   - Imputed missing values for key numerical features (`host_listings_count`, `bedrooms`, etc.) using column means
   - Created missingness indicator variables for imputed columns

3. **Feature Engineering**:
   - One-hot encoded categorical variables (`host_response_time`, `room_type`)
   - Processed `neighbourhood_group_cleansed` into 5 borough categories
   - Identified unstructured text fields for potential NLP processing

4. **Exploratory Analysis**:
   - Calculated feature correlations with price label
   - Visualized relationships using Seaborn pairplots
   - Identified `accommodates` (r=0.50) and `bedrooms` (r=0.42) as top correlates

## Key Findings
- The dataset required significant cleaning with 23 columns containing missing values
- Winsorizing price values removed extreme outliers while preserving distribution shape
- Text fields like `description` and `neighborhood_overview` contain valuable unstructured data
- Moderate correlation exists between listing size features and price

## Files Included
1. **`airbnbData.csv`**: Raw dataset with 38,277 listings
2. **`airbnbData_cleaned.csv`**: Processed dataset after cleaning (generated during lab)

## How to Reproduce
1. Load the raw dataset and inspect data types/missingness
2. Apply winsorization to the price column
3. Impute missing values for selected numerical features
4. Perform one-hot encoding on categorical variables
5. Generate correlation analysis and visualizations

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `scipy.stats`, `matplotlib`, `seaborn`

## Usage
To reuse the data cleaning pipeline:
```python
from scipy.stats import mstats

# Winsorize prices
df['label_price'] = mstats.winsorize(df['price'], limits=[0.01, 0.01])

# Impute missing values
for col in ['bedrooms', 'bathrooms']:
    df[col].fillna(df[col].mean(), inplace=True)
    
# One-hot encode categoricals
df = pd.get_dummies(df, columns=['room_type'])
```

## Key Visualizations
*Relationship between price, accommodates and bedrooms*
```

This maintains the same professional structure as your Lab 5 README while adapting the content to Lab 1's focus on data preparation. Key differences include:
1. Emphasis on data cleaning rather than model evaluation
2. Inclusion of specific data issues addressed (missing values, outliers)
3. Code examples for core cleaning operations
4. Reference to generated visualizations
5. Clear pipeline reproduction steps

The format ensures consistency across all lab documentation while properly representing each lab's unique focus within the ML lifecycle.
