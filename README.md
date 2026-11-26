# Credit Card Fraud Detection 💳

## About
This project uses Machine Learning to detect fraudulent credit card transactions.

It uses LightGBM to classify transactions into two categories:
- 0 → Genuine Transaction
- 1 → Fraudulent Transaction

The project uses probability-based predictions and custom classification thresholds to adjust fraud detection sensitivity.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- MinMaxScaler
- Joblib

## Dataset Features

The dataset contains 30 input features:

- `Time` – Time elapsed since the first transaction.
- `V1` to `V28` – PCA-transformed numerical features.
- `Amount` – Transaction amount.
- `Class` – Target variable (0 or 1).

## Model Workflow
1. Load the credit card dataset.
2. Separate features and target.
3. Split the dataset into training and testing sets.
4. Preprocess numerical features.
5. Train a LightGBM classifier.
6. Evaluate using Precision, Recall, F1-score and Confusion Matrix.
7. Experiment with different classification thresholds.
8. Save the trained model.
9. Predict fraudulent transactions.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

## Example Prediction

```text
================================
    CREDIT CARD FRAUD DETECTION
================================

Fraud Probability: 85.42 %
Threshold: 0.15

Result: FRAUDULENT TRANSACTION
```

*Example output only.*

## Important Note
The dataset is highly imbalanced, meaning genuine transactions greatly outnumber fraudulent transactions.

A custom classification threshold is used to adjust fraud detection sensitivity.

This project is for educational purposes and is not intended for real-world financial decisions.

## Future Improvements
- Improve fraud detection performance.
- Experiment with different thresholds.
- Compare LightGBM with XGBoost.
- Build a web interface using Flask.

## Author
Sayak Bose

Machine Learning Enthusiast
