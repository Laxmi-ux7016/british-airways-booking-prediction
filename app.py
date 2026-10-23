
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="British Airways AI", page_icon="✈️", layout="wide")

st.title("✈️ British Airways Booking Prediction")
st.write("Explore booking data and predict booking completion.")

file = st.file_uploader("Upload your CSV dataset", type="csv")

if file:
    df = pd.read_csv(file, encoding="latin1")

    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(), use_container_width=True)

    column = st.selectbox("Select column for visualization", df.columns)
    st.plotly_chart(
        px.histogram(df, x=column, title=f"{column} Distribution"),
        use_container_width=True
    )

    target = st.selectbox("Select target column", df.columns)

    if st.button("🚀 Train Model"):
        data = df.dropna(subset=[target]).copy()
        X = data.drop(columns=[target])
        y = data[target]

        X = pd.get_dummies(X).fillna(0)
        y = y.astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model Accuracy: {accuracy:.2%}")

        st.session_state["model"] = model
        st.session_state["columns"] = X.columns.tolist()

    if "model" in st.session_state:
        st.info("Model trained successfully!")
        st.write("Target:", target)
