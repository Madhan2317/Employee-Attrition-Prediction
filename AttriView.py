import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import joblib

# Load model and scaler

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbNSHgrhu-q6keTki5jYV75LC70MjPSnzx2w&s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;    
    }
    .main {
        background-color: reduced transparency from 0.7 to 0 
        padding: 2rem;
        border-radius: 10px;
    }
     .block-container {
        background-color: rgba(30,25,40, 0.75);
        padding: 2rem;
        border-radius: 12px;
        layout: wide;
    }
    </style>
""", unsafe_allow_html=True)

# Set the page config for wider layout (optional)
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: #d1ac34;'>🚓 Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color: #d1ac34;'>An interactive dashboard for predicting employee attrition.</p>", unsafe_allow_html=True)

# Input fields for employee attributes
st.markdown("<h2 style='color: #d1ac34;'>Enter Employee Details</h2>", unsafe_allow_html=True)

# Create a wider layout with padding
col1, col2 = st.columns([1.5, 1.5], gap="large")

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    

with col2:
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
    overtime = st.selectbox("OverTime", ['Yes', 'No'])
    percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=3)
    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
    years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)

# Derived features
tenure_per_job_level = years_at_company / (job_level + 1)
promotion_lag = years_since_last_promotion / (years_at_company + 1)

# Collect all inputs into a DataFrame
input_dict = {
    'Age': age,
    'BusinessTravel': business_travel,
    'Department': department,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'EducationField': education_field,
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender': gender,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'OverTime': overtime,
    'PercentSalaryHike': percent_salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times_last_year,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_current_role,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'YearsWithCurrManager': years_with_curr_manager,
    'TenurePerJobLevel': tenure_per_job_level,
    'PromotionLag': promotion_lag
}

input_df = pd.DataFrame([input_dict])

# Encoding categorical features
def encode_inputs(df):
    mappings = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
        'EducationField': {'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Other': 4, 'Human Resources': 5},
        'Gender': {'Male': 1, 'Female': 0},
        'JobRole': {
            'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
            'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
            'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8
        },
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'OverTime': {'Yes': 1, 'No': 0}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

input_encoded = encode_inputs(input_df)

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.markdown(
    """
    <div style='background-color:#d1e7dd;padding:15px;border-radius:10px;color:#0f5132;font-size:18px;'>
        ✅ <strong>Prediction:</strong> This employee is likely to <strong>stay</strong> with the organization.
    </div>
    """,
    unsafe_allow_html=True
)

    else:
        st.markdown(
    """
    <div style='background-color:#f8d7da;padding:15px;border-radius:10px;color:#842029;font-size:18px;'>
        ⚠️ <strong>Prediction:</strong> This employee is at a high <strong>risk of leaving</strong> the organization.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("## 📊 Insights from Employee Data")
st.markdown("---")

# Load your original dataset
df = pd.read_csv("D:\MDTE21\Employee Attrition\Employee-Attrition - Employee-Attrition.csv")

# Create 2 columns
col1, col2 = st.columns(2)

# Line Chart — e.g., Monthly Income over Age
with col1:
    st.markdown("#### 📈 Line Chart: Income vs Age")
    fig_line = px.line(
        df.sort_values("Age"),
        x="Age",y="MonthlyIncome",color="Attrition",
        title="Monthly Income by Age",
        color_discrete_map={
            "Yes": "#FF6B6B",  # Red for employees who left
            "No": "#1FAB89"    # Green for employees who stayed
        }
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Bar Chart — e.g., Attrition by Job Role
with col2:
    st.markdown("#### 📊 Clustered Column Chart: Attrition by Job Role")
    attrition_jobrole = df.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
    fig_cluster = px.bar(
        attrition_jobrole,
        x="JobRole",y="Count",color="Attrition",
        barmode="group",  # Grouped (side-by-side)
        title="Attrition Count by Job Role (Clustered)",
        color_discrete_map={
            "Yes": "#FF6B6B",  # Red/pink shade for attrition
            "No": "#1FAB89"    # Green/teal shade for retention
        }
    )
    fig_cluster.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cluster, use_container_width=True)

# Next row
col3, col4 = st.columns(2)

# Pie Chart — e.g., Attrition Rate
with col3:
    st.markdown("#### 🥧 Pie Chart: Attrition Ratio")
    attr_counts = df["Attrition"].value_counts().reset_index()
    attr_counts.columns = ["Attrition", "Count"]
    fig_pie = px.pie(
        attr_counts,names="Attrition",values="Count",
        hole=0.3,title="Employee Attrition Distribution",
        color_discrete_map={
            "Yes": "#FF6B6B",  # Red for employees who left
            "No": "#1FAB89"    # Green for employees who stayed
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Histogram — e.g., Age Distribution
with col4:
    st.markdown("#### 📊 Histogram: Age Distribution")
    fig_hist = px.histogram(
        df, x="Age",color="Attrition",
        barmode="overlay",title="Age Distribution by Attrition Status",
        opacity=0.7,
        color_discrete_map={
            "Yes": "#2A09E5",  # Dark gray for employees who left
            "No": "#00CC96"    # Green for employees who stayed
        }
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
# Display the first few rows of the dataset
st.markdown("### 📋 Dataset Preview")
st.dataframe(df.head())
st.markdown("---")