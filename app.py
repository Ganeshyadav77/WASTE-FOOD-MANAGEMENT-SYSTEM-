import streamlit as st
import pandas as pd
from eda import show_eda
from model import load_model, make_prediction

# Page configuration
st.set_page_config(
    page_title="Data Analytics & Prediction Platform",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Data Overview", "Exploratory Data Analysis", "Model Prediction"]
)

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")
    return df

df = load_data()

# Render selected page
if page == "Data Overview":
    st.title("📋 Data Overview")
    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.dataframe(df.dtypes.rename("Data Type"))
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Sample Data (First 100 rows)")
    st.dataframe(df.head(100))

elif page == "Exploratory Data Analysis":
    show_eda(df)

elif page == "Model Prediction":
    st.title("🎯 Model Prediction")
    
    # Load pre-trained model
    model = load_model("models/model.pkl")
    
    st.subheader("Enter Feature Values")
    
    # Dynamic input generation based on your features
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.number_input("Feature 1", value=0.0, step=0.1)
        feature2 = st.number_input("Feature 2", value=0.0, step=0.1)
    with col2:
        feature3 = st.number_input("Feature 3", value=0.0, step=0.1)
        feature4 = st.number_input("Feature 4", value=0.0, step=0.1)
    
    if st.button("🚀 Predict", type="primary"):
        features = [[feature1, feature2, feature3, feature4]]
        prediction, probability = make_prediction(model, features)
        
        st.success(f"**Prediction Result:** {prediction[0]}")
        if probability is not None:
            st.metric("Confidence Score", f"{probability.max():.2%}")
