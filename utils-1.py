import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess data with caching"""
    df = pd.read_csv(file_path)
    return df

def preprocess_features(df, target_col=None):
    """Separate features and target, encode categorical variables"""
    X = df.copy()

    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(columns=[target_col])
    else:
        y = None

    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split data and apply standardization"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
