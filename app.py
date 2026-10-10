import streamlit as st
import pandas as pd
import joblib

# Load the retrained model
model = joblib.load("best_model.pkl")

st.title("Retail Sales Prediction")

# Input fields
price = st.number_input("Enter Price", min_value=0.0, value=20.0)

promo = st.selectbox("Promotion", [0, 1])

weekday = st.selectbox("Weekday", [1, 2, 3, 4, 5, 6, 7])

month = st.selectbox("Month", list(range(1, 13)))

store = st.selectbox(
    "Select Store",
    [f"store_{i}" for i in range(1, 11)]
)

if st.button("Predict Sales"):

    # Prepare input data
    data = pd.DataFrame({
        "price": [price],
        "promo": [promo],
        "weekday": [weekday],
        "month": [month],
        "store_id": [store]
    })

    # Convert store categories into numerical columns
    # if the model was trained with one-hot encoding.
    data = pd.get_dummies(data, columns=["store_id"])

    # Match the features expected by the trained model
    data = data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(data)

    st.success(f"Predicted Sales: {round(prediction[0])} units")