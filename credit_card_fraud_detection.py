# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
credit_card_data = pd.read_csv(r"C:\Users\ishik\OneDrive\Desktop\Credit Card\creditcard.csv")

# Separate legit and fraud cases
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Undersample legit data
legit_sample = legit.sample(n=150, random_state=42)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Features and labels
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy evaluation
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

# Streamlit app
st.title("💳 Credit Card Fraud Detection App")

st.markdown("### Model Accuracy")
st.write(f"Training Accuracy: **{train_accuracy:.2f}**")
st.write(f"Test Accuracy: **{test_accuracy:.2f}**")

st.markdown("---")
st.markdown("### Predict a Transaction")
input_text = st.text_input("Enter 30 comma-separated feature values (excluding Class):")

if st.button("Submit"):
    try:
        input_values = np.array([float(x) for x in input_text.split(',')])
        if len(input_values) != 30:
            st.error("❌ Please enter exactly 30 values.")
        else:
            input_scaled = scaler.transform(input_values.reshape(1, -1))
            prediction = model.predict(input_scaled)
            result = "✅ Legitimate Transaction" if prediction[0] == 0 else "🚨 Fraudulent Transaction"
            st.success(result)
    except:
        st.error("⚠️ Invalid input. Please enter numeric values only.")

