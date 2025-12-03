import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Earthquake Clustering Prediction App")
st.write("Masukkan koordinat untuk memprediksi cluster gempa.")

# Input user
latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)

if st.button("Prediksi Cluster"):
    data = pd.DataFrame([[latitude, longitude]], columns=["latitude", "longitude"])
    
    try:
        pred = model.predict(data)[0]
        st.success(f"Cluster hasil prediksi: **{pred}**")
    except Exception as e:
        st.error("Terjadi error saat memprediksi.")
        st.write(e)
