import streamlit as st
import pandas as pd
import cloudpickle

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

st.title("Earthquake Cluster Prediction")

lat = st.number_input("Latitude", value=0.0)
lon = st.number_input("Longitude", value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([[lat, lon]], columns=["latitude", "longitude"])
    pred = model.predict(df)[0]
    st.success(f"Predicted Cluster: {pred}")
