import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the saved model and dataset
model = pickle.load(open(r'F:\vscode\EMPLOYEE ATTRITION\random_forest_model.pkl', 'rb'))
data = pd.read_csv(r'F:\vscode\EMPLOYEE ATTRITION\resampled_with_target.csv')

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predict Employee Attrition"])

if page == "🏠 Home":
    st.title("High-Risk Employee List")
    st.write("Employees with highest likelihood of attrition based on model prediction.")

    # Simulate predictions to show high-risk employees
    data['Prediction'] = model.predict(data.drop('Attrition', axis=1))
    high_risk = data[data['Prediction'] == 1]

    st.dataframe(high_risk.head(10))  # Show top 10 high-risk employees

    st.title("Job Satisfaction & Performance Insights")
    st.write("Average values for key features by attrition class:")

    if 'jobsatisfaction' in data.columns and 'performanceRating' in data.columns:
        insights = data.groupby('Attrition')[['jobsatisfaction', 'performanceRating']].mean()
        st.dataframe(insights)
    else:
        st.write("Job satisfaction or performance rating columns not found.")

elif page == "🔮 Predict Employee Attrition":
    st.title("Predict Employee Attrition")
    st.write("Enter employee details to predict if they will leave or stay.")

    # Define feature inputs
    inputs = {}
    for col in data.columns:
        if col not in ['Attrition', 'Prediction']:
            inputs[col] = st.number_input(f"{col.capitalize()}", value=float(data[col].mean()))

    if st.button("Predict Attrition"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        result = "Yes (Left)" if prediction == 1 else "No (Stayed)"
        st.success(f"Predicted Attrition: {result}")
