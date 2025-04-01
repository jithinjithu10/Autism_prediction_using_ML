# Autism Spectrum Disorder (ASD) Classification

## Welcome to the Autism Spectrum Disorder (ASD) Classification Project!
This machine learning project leverages Python to predict ASD traits based on survey data, achieving 82% accuracy. It involves data preparation, model testing, and real-world application to support early ASD detection.

## Project Overview
This project classifies ASD using survey responses, including AQ-10 scores, age, and gender, to predict labels (0 for no ASD, 1 for ASD). Multiple models were tested, and Random Forest emerged as the most effective.

## Contents
- **ASD_Classification.py**: Main Python script with the full implementation.
- **train.csv**: Dataset containing 800 rows and 22 feature columns along with ASD labels.
- **best_model.pkl**: Saved Random Forest model for predictions.

## How It Works
### Step-by-step breakdown:
1. **Loads Tools**: Uses `pandas`, `scikit-learn`, `xgboost`, and other libraries for data processing and modeling.
2. **Gets Data**: Reads `train.csv`, which contains survey responses, demographics, and ASD labels.
3. **Preps Data**: Converts categorical data to numerical using `LabelEncoder`, balances data using `SMOTE`, and splits it into training/testing sets.
4. **Tests Models**: Runs Decision Tree, Random Forest, and XGBoost models with hyperparameter tuning.
5. **Selects Winner**: Random Forest achieves 93% cross-validation accuracy and is saved as `best_model.pkl`.
6. **Shows Results**: Achieves 82% accuracy on the test set—correctly predicting 108 "no ASD" cases and 23 "ASD" cases, with some misclassifications.

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imblearn xgboost
```

## Running the Project
1. Clone the repository:
```bash
git clone https://github.com/your-username/ASD-Classification.git
cd ASD-Classification
```
2. Ensure `train.csv` and `ASD_Classification.py` are present in the folder.
3. Execute the script:
```bash
python ASD_Classification.py
```
4. View the output—accuracy, confusion matrix, and additional metrics.

## Results
- **Accuracy**: 82% on the test set.
- **Confusion Matrix**:
  ```
  [[108  16]
   [ 13  23]]
  ```
  - 108 correct "no ASD"
  - 23 correct "ASD"
- **Best Model**: Random Forest (`max_depth=20, n_estimators=50`)

## Future Steps
- Implement real-time predictions for new data.
- Explore feature engineering to improve accuracy.

## Purpose
This project demonstrates how data science can address real-world challenges, such as early ASD detection, using practical machine learning techniques. By refining these methods, we can enhance diagnostic support and awareness for ASD.

---
