import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from xgboost import XGBClassifier

# Load necessary components
# Load the pre-trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load the pre-trained transformer for feature preprocessing
with open("transformer.pkl", "rb") as f:
    transformer = pickle.load(f)

# Define all expected columns based on the training dataset
def get_expected_columns():
    return [
        "Transaction Amount", "Transaction Hour", "Product Category",
        "Quantity", "Device Used", "Is Address Match", "Transaction DOW",
        "Transaction Day", "Transaction Month", "Payment Method",
        "Account Age Days", "Customer Age"
    ]

# Function to preprocess user input to match the model's expected input format
def preprocess_input(data, transformer):
    """
    Ensure the input data matches the transformer's expected structure by:
    - Adding missing columns with default values
    - Ordering columns correctly
    """
    expected_columns = get_expected_columns()

    # Add missing columns with default values
    for col in expected_columns:
        if col not in data.columns:
            if col in ["Transaction Amount", "Account Age Days", "Customer Age"]:
                data[col] = 0.0  # Default for numeric values
            elif col in ["Transaction Hour", "Transaction DOW", "Transaction Day", "Transaction Month", "Quantity"]:
                data[col] = 0  # Default for integer values
            else:
                data[col] = "Unknown"  # Default for categorical columns

    # Ensure column order matches the transformer's expectations
    data = data[expected_columns]

    # Apply the transformer to preprocess the data
    return transformer.transform(data)

# Main Streamlit app
def main():
    # Set app title and description
    st.title("Fraudify")
    st.markdown("Protecting Every Transaction, Every Time!")

    # Section: Input transaction details
    st.subheader("Transaction Details:")

    # Input fields for user to provide transaction details
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
    transaction_hour = st.number_input("Transaction Hour (0-23) (e.g., 8.30am = 8)", min_value=0, max_value=23, step=1)
    product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home & Garden", "Toys", "Others"])
    quantity = st.number_input("Product Quantity", min_value=1, step=1)
    device_used = st.selectbox("Device Used", ["Mobile", "Laptop", "Tablet", "Desktop"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Others"])
    is_address_match = st.selectbox("Is the Billing Address Same as the Shipping Address?", ["Yes", "No"])

    # Derived features
    transaction_day = transaction_date.day  # Day of the month
    transaction_month = transaction_date.month  # Month of the year
    transaction_dow = transaction_date.weekday()  # Day of the week (0=Monday, 6=Sunday)

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        "Transaction Amount": [transaction_amount],
        "Transaction Hour": [transaction_hour],
        "Product Category": [product_category],
        "Quantity": [quantity],
        "Device Used": [device_used],
        "Is Address Match": [1 if is_address_match == "Yes" else 0],
        "Transaction DOW": [transaction_dow],
        "Transaction Day": [transaction_day],
        "Transaction Month": [transaction_month],
        "Payment Method": [payment_method]
    })

    # Preprocess the input and make a prediction
    try:
        # Preprocess the input to match model requirements
        preprocessed_input = preprocess_input(user_input, transformer)

        # Button to trigger fraud prediction
        if st.button("Predict Fraud"):
            # Get prediction from the model
            prediction = xgb_model.predict(preprocessed_input)[0]

            # Display prediction results
            if prediction == 1:
                st.error("⚠️ This transaction is predicted to be FRAUDULENT!")
            else:
                st.success("✅ This transaction is predicted to be SAFE.")
    except ValueError as e:
        st.error(f"Error in processing input: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
