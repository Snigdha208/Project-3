
import streamlit as st
import pandas as pd
import pickle

# Load model and dataset
model = pickle.load(open(r'F:\vscode\EMPLOYEE ATTRITION\xgboost_model.pkl', 'rb'))
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

    if 'jobsatisfaction' in data.columns and 'performancerating' in data.columns:
        insights = data.groupby('Attrition')[['jobsatisfaction', 'performancerating']].mean()
        st.dataframe(insights)
    else:
        st.write("Job satisfaction or performance rating columns not found.")

elif page == "🔮 Predict Employee Attrition":
    st.title("Predict Employee Attrition")
    st.write("Enter employee details to predict if they will leave or stay.")

    # Allow user to select an EmployeeNumber to auto-fill form
    employee_ids = data['employeenumber'].unique()
    selected_id = st.selectbox("Select EmployeeNumber to auto-fill inputs", employee_ids)

    selected_row = data[data['employeenumber'] == selected_id].iloc[0]

    # Input form using values from the selected row
    inputs = {}
    for col in data.columns:
        if col not in ['Attrition', 'Prediction']:
            default_value = float(selected_row[col])
            inputs[col] = st.number_input(f"{col.capitalize()}", value=default_value)

    if st.button("Predict Attrition"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        result = "Yes (Left)" if prediction == 1 else "No (Stayed)"
        st.success(f"Predicted Attrition: {result}")


# No - 0 (employee stayed)
# Yes - 1 (employee Left)       
