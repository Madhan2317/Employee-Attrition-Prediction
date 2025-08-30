import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load saved files
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Prediction function
def predict_attrition(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Encode categoricals
    for col in encoders:
        if col in input_df.columns:
            le = encoders[col]
            input_df[col] = le.transform([input_df[col].values[0]])[0]
    
    # Feature engineering
    input_df['TenurePerJobLevel'] = input_df['YearsAtCompany'] / (input_df['JobLevel'] + 1)
    input_df['PromotionLag'] = input_df['YearsSinceLastPromotion'] / (input_df['YearsAtCompany'] + 1)
    
    # Ensure column order matches training
    training_columns = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 
                        'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 
                        'MaritalStatus', 'MonthlyIncome', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
                        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                        'YearsWithCurrManager', 'TenurePerJobLevel', 'PromotionLag']
    input_df = input_df[training_columns]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    proba = model.predict_proba(input_scaled)[:, 1][0]
    pred = 1 if proba >= 0.5 else 0
    return pred, proba

# CSS for styling with background image
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbNSHgrhu-q6keTki5jYV75LC70MjPSnzx2w&s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 2rem;
    }
    .block-container {
        background-color: rgba(30, 25, 40, 0.75);
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #d1ac34;
    }
    .stMarkdown, .stText, .stSelectbox, .stNumberInput {
        color: white;
    }
    .stButton>button {
        background-color: #d1ac34;
        color: #2c3e50;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #b58900;
    }
    </style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🚓 Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>An interactive tool for predicting employee turnover using Random Forest and analyzing trends.</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Model Evaluation", "Predict Attrition"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Visualizations of key trends and factors influencing employee attrition.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('attrition_by_department.png', caption='Attrition by Department', use_container_width=True)
        st.image('job_satisfaction_vs_attrition.png', caption='Job Satisfaction vs Attrition', use_container_width=True)
    with col2:
        st.image('correlation_heatmap.png', caption='Correlation Heatmap', use_container_width=True)
    
    # Interactive plot
    st.markdown("#### Age Distribution by Attrition")
    df = pd.read_csv("D:\MDTE21\Employee Attrition\Employee-Attrition - Employee-Attrition.csv")  # Update path
    fig_hist = px.histogram(
        df, x="Age", color="Attrition",
        barmode="overlay", title="Age Distribution by Attrition Status",
        opacity=0.7, color_discrete_map={"Yes": "#FF6B6B", "No": "#00CC96"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("Model Evaluation")
    st.markdown("Performance metrics for the Random Forest model.")
    
    st.subheader("Random Forest")
    # Replace with your actual metrics from training output
    st.text("Best Params: {'n_estimators': 200, 'max_depth': None}")  # Example
    st.text("Accuracy: 0.95")  # Example
    st.text("Precision: 0.94")
    st.text("Recall: 0.96")
    st.text("F1 Score: 0.95")
    st.text("ROC AUC: 0.98")
    st.text("Confusion Matrix:\n[[310, 10], [15, 290]]")
    st.text("Classification Report:\n              precision    recall  f1-score\n0       0.95      0.97      0.96\n1       0.97      0.95      0.96")

with tab3:
    st.header("Predict Attrition for a New Employee")
    st.markdown("Enter employee details to predict their likelihood of leaving using the Random Forest model.")
    
    input_data = {}
    
    with st.form("employee_form"):
        col1, col2 = st.columns([1.5, 1.5], gap="large")
        
        with col1:
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=60, value=30)
            input_data['BusinessTravel'] = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
            input_data['Department'] = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
            input_data['DistanceFromHome'] = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
            input_data['Education'] = st.selectbox("Education", [1, 2, 3, 4, 5])
            input_data['EducationField'] = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
            input_data['EnvironmentSatisfaction'] = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
            input_data['JobInvolvement'] = st.selectbox("Job Involvement", [1, 2, 3, 4])
            input_data['JobLevel'] = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            input_data['JobRole'] = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
            input_data['JobSatisfaction'] = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            input_data['MaritalStatus'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        
        with col2:
            input_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
            input_data['OverTime'] = st.selectbox("OverTime", ['Yes', 'No'])
            input_data['PercentSalaryHike'] = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
            input_data['PerformanceRating'] = st.selectbox("Performance Rating", [1, 2, 3, 4])
            input_data['RelationshipSatisfaction'] = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
            input_data['StockOptionLevel'] = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            input_data['TotalWorkingYears'] = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
            input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
            input_data['WorkLifeBalance'] = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            input_data['YearsAtCompany'] = st.number_input("Years At Company", min_value=0, max_value=40, value=3)
            input_data['YearsInCurrentRole'] = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
            input_data['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
            input_data['YearsWithCurrManager'] = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        pred, proba = predict_attrition(input_data)
        if pred == 1:
            st.error(f"The employee is at high risk of attrition (probability: {proba:.2%}). Consider retention strategies like career development or improved compensation.")
        else:
            st.success(f"The employee is likely to stay (probability of staying: {1 - proba:.2%}).")
    
    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

# Footer
st.markdown("---")
st.markdown("Developed for HR Analytics | Powered by Streamlit")