
import pandas as pd
import joblib

data = joblib.load("credit_card.pkl")

model = data["model"]
threshold = data["threshold"]

print("\n================================")
print("    CREDIT CARD FRAUD DETECTION")
print("================================")

time = float(input("Enter transaction time: "))
amount = float(input("Enter transaction amount: "))

features = {}

features["Time"] = time

for i in range(1, 29):
    features[f"V{i}"] = float(input(f"Enter V{i}: "))

features["Amount"] = amount

new_data = pd.DataFrame([features])

probability = model.predict_proba(new_data)[0][1]

prediction = int(probability >= threshold)

print("\n================================")
print("       PREDICTION RESULT")
print("================================")

print("Fraud Probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Result: FRAUDULENT TRANSACTION")
else:
    print("Result: GENUINE TRANSACTION")

print("================================")
