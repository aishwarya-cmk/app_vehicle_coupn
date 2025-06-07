import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Streamlit page config
st.set_page_config(page_title="Coupon Prediction", layout="wide")
st.title("🚗 In-Vehicle Coupon Recommendation (Logistic Regression)")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    # Drop missing values
    df.dropna(inplace=True)

    # Target column check
    if 'Y' not in df.columns:
        st.error("❌ The dataset must contain a target column named 'Y'.")
        st.stop()

    # Label Encode Target
    df['Y'] = LabelEncoder().fit_transform(df['Y'])

    # Required features
    features = ['destination', 'passenger', 'weather', 'temperature', 'time', 'coupon',
                'gender', 'age', 'maritalStatus', 'has_children', 'education',
                'occupation', 'income', 'car']

    available = [f for f in features if f in df.columns]
    st.info(f"✅ Found features: {available}")

    # Encode categorical features
    label_encoders = {}
    for col in available:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Filter dataset
    df = df[available + ['Y']]

    # Class balance before
    st.subheader("⚖️ Class Distribution (Before Balancing)")
    st.bar_chart(df['Y'].value_counts())

    # Balance classes
    df_major = df[df.Y == 0]
    df_minor = df[df.Y == 1]
    df_minor_upsampled = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    df_balanced = pd.concat([df_major, df_minor_upsampled])

    st.subheader("⚖️ Class Distribution (After Balancing)")
    st.bar_chart(df_balanced['Y'].value_counts())

    # Train model
    X = df_balanced.drop('Y', axis=1)
    y = df_balanced['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Results
    st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.subheader("📊 Classification Report")
    st.code(classification_report(y_test, y_pred))

    st.subheader("📌 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("📈 Feature Importance")
    fig_feat, ax_feat = plt.subplots()
    ax_feat.barh(X.columns, model.coef_[0])
    ax_feat.set_xlabel("Coefficient")
    st.pyplot(fig_feat)

    st.subheader("🔍 Test Predictions")
    results_df = X_test.copy()
    results_df["Actual"] = y_test
    results_df["Predicted"] = y_pred
    st.dataframe(results_df.head(10))

    # Sidebar for user input
    st.sidebar.header("🔮 Try Your Own Input")
    user_input = {}
    for col in available:
        if col in label_encoders:
            options = label_encoders[col].classes_
            selected = st.sidebar.selectbox(f"{col}", options)
            user_input[col] = label_encoders[col].transform([selected])[0]
        else:
            val = st.sidebar.slider(f"{col}", int(df[col].min()), int(df[col].max()))
            user_input[col] = val

    input_df = pd.DataFrame([user_input])
    user_pred = model.predict(input_df)[0]
    user_prob = model.predict_proba(input_df)[0][user_pred]

    st.sidebar.subheader("🎯 Prediction")
    result = "✅ Will Accept the Coupon" if user_pred == 1 else "❌ Will NOT Accept the Coupon"
    st.sidebar.success(f"{result} ({user_prob*100:.2f}% confidence)")

else:
    st.info("📥 Please upload a CSV file to get started.")
