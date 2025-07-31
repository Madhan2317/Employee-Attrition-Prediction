Employee Attrition Prediction using Streamlit
=============================================

📌 Overview
Employee Attrition Prediction is a Streamlit-based interactive web application that helps HR professionals and data analysts predict whether an employee is likely to stay or leave the company. Using a trained machine learning model, the app takes employee-specific information as input and predicts the attrition outcome.

🛠️ Tech Stack Used
- Python — Core language
- Streamlit — For building interactive UI
- scikit-learn — For machine learning modeling
- Pandas & NumPy — Data preprocessing and manipulation
- Plotly — For creating dynamic and interactive visualizations
- Joblib / Pickle — Model serialization and loading
- Visual Studio Code— For model development and analysis

🚀 Features
- 📄 Clean and responsive Streamlit UI with two-column form layout  
- 📊 Visual insights including:
  - Line chart (Age vs Monthly Income)
  - Bar chart (Attrition by Department)
  - Pie chart (Attrition Distribution)
  - Scatter plot (Age vs Years at Company)
  - Clustered column charts for department, education field, etc.
- 🔍 Prediction of employee status: "Will Stay" or "Will Leave"
- 📥 Sample inputs for testing
- 🧠 Model trained on real-world HR dataset with 28 features
- 📝 Inline explanations and tooltips for better user understanding

🔍 How It Works:
----------------
1. User fills out employee features in the form (e.g., age, gender, department, etc.)
2. Inputs are encoded and scaled using the same pipeline as training
3. Model predicts attrition risk and returns:
   - “Prediction: Employee will Stay” ✅
   - “Prediction: Employee will Leave” ❌
4. Data visualizations provide key HR insights

📊 Insights:
------------
- Employees with high overtime are more likely to leave
- Job roles like "Sales Executive" have higher attrition
- Younger employees with lower income tend to leave more often
- Longer tenure and higher satisfaction correlate with retention

📊 Sample Input
| Feature              | Example 1 (Stay) | Example 2 (Leave) |
|----------------------|------------------|-------------------|
| Age                  | 29               | 42                |
| MonthlyIncome        | 7000             | 4500              |
| JobSatisfaction      | 4                | 1                 |
| OverTime             | No               | Yes               |


📃 License
This project is licensed under the MIT License – feel free to use and modify.
