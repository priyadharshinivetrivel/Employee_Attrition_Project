import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load Files
df = pd.read_csv("Palo Alto Networks (1).csv")
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")



# Title
st.title("🏢 Employee Attrition Prediction & Risk Scoring System")

st.success("✅ Model Loaded Successfully!")

# ==========================
# Dashboard Metrics
# ==========================

total_employees = len(df)
employees_left = df["Attrition"].sum()
employees_stayed = total_employees - employees_left
attrition_rate = (employees_left / total_employees) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("👨‍💼 Total Employees", total_employees)
col2.metric("🚪 Employees Left", employees_left)
col3.metric("😊 Employees Stayed", employees_stayed)
col4.metric("📈 Attrition Rate", f"{attrition_rate:.2f}%")

st.divider()

st.subheader("📊 Employee Attrition Distribution")

attrition_counts = df["Attrition"].value_counts()

chart_data = pd.DataFrame({
    "Status": ["Stayed", "Left"],
    "Count": [attrition_counts[0], attrition_counts[1]]
})

st.bar_chart(chart_data.set_index("Status"))

# ==========================
# Sidebar
# ==========================

st.sidebar.title("🔍 Filters")

department = st.sidebar.selectbox(
    "Select Department",
    ["All"] + sorted(df["Department"].unique().tolist())
)

if department != "All":
    filtered_df = df[df["Department"] == department]
else:
    filtered_df = df

st.subheader("📋 Filtered Employee Data")

st.dataframe(filtered_df)

st.subheader("🏢 Department Summary")

dept_summary = (
    filtered_df.groupby("Department")
    .agg(
        Employees=("Department", "count"),
        Attrition=("Attrition", "sum")
    )
)

st.dataframe(dept_summary)

st.subheader("👤 Employee Profile")

employee_index = st.selectbox(
    "Select Employee",
    filtered_df.index
)

employee = filtered_df.loc[employee_index]

st.subheader("🤖 Employee Attrition Prediction")

if st.button("Predict Employee Risk"):

    # Create DataFrame from selected employee
    employee_df = pd.DataFrame([employee])

    # Remove target column if present
    if "Attrition" in employee_df.columns:
        employee_df = employee_df.drop(columns=["Attrition"])

    # One-Hot Encoding
    employee_df = pd.get_dummies(employee_df)

    # Add missing columns
    for col in feature_columns:
        if col not in employee_df.columns:
            employee_df[col] = 0

    # Arrange columns in same order
    employee_df = employee_df[feature_columns]

    # Scale
    employee_scaled = scaler.transform(employee_df)

    # Prediction
    prediction = model.predict(employee_scaled)[0]
    probability = model.predict_proba(employee_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.write(f"**Probability of Attrition:** {probability:.2%}")

    if probability < 0.30:
        st.success("🟢 Low Risk")

    elif probability < 0.60:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")

    st.subheader("💡 Possible Risk Factors")

    if employee["OverTime"] == "Yes":
        st.write("✔ Employee works overtime")

    if employee["JobSatisfaction"] <= 2:
        st.write("✔ Low job satisfaction")

    if employee["WorkLifeBalance"] <= 2:
        st.write("✔ Poor work-life balance")

    if employee["YearsSinceLastPromotion"] >= 5:
        st.write("✔ Long time since last promotion")

st.write("### Employee Information")

st.subheader("📈 Top 10 Important Features")

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

st.bar_chart(feature_importance.set_index("Feature"))

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Age:** {employee['Age']}")
    st.write(f"**Department:** {employee['Department']}")
    st.write(f"**Job Role:** {employee['JobRole']}")
    st.write(f"**Monthly Income:** ₹{employee['MonthlyIncome']}")

with col2:
    st.write(f"**OverTime:** {employee['OverTime']}")
    st.write(f"**Job Satisfaction:** {employee['JobSatisfaction']}")
    st.write(f"**Work Life Balance:** {employee['WorkLifeBalance']}")
    st.write(f"**Years At Company:** {employee['YearsAtCompany']}")

    st.subheader("📊 Department-wise Attrition")

dept_attrition = (
    df.groupby("Department")["Attrition"]
      .sum()
      .reset_index()
)

st.bar_chart(
    dept_attrition.set_index("Department")
)

st.subheader("⏰ Overtime Analysis")

overtime = (
    df.groupby("OverTime")["Attrition"]
      .sum()
      .reset_index()
)

st.bar_chart(
    overtime.set_index("OverTime")
)

st.subheader("😊 Job Satisfaction vs Attrition")

job = (
    df.groupby("JobSatisfaction")["Attrition"]
      .sum()
      .reset_index()
)

st.line_chart(
    job.set_index("JobSatisfaction")
)

st.sidebar.subheader("⚙ Risk Threshold")

risk_threshold = st.sidebar.slider(
    "Select Risk Threshold",
    0.0,
    1.0,
    0.60
)

st.write("Current Risk Threshold:", risk_threshold)



st.download_button(
    label="📥 Download Employee Data",
    data=df.to_csv(index=False),
    file_name="employee_data.csv",
    mime="text/csv"
)

page = st.sidebar.radio(
    "📂 Navigation",
    [
        "Dashboard",
        "Employee Prediction",
        "Department Analysis",
        "Feature Importance",
        "About Project"
    ]
)
if page == "Dashboard":
    ...

elif page == "Employee Prediction":
    ...

elif page == "Department Analysis":
    ...

st.markdown("---")

st.caption(
    "Machine Learning-Based Employee Attrition Prediction and Risk Scoring System | Unified Mentor | Palo Alto Networks"
)

st.sidebar.info("""
This application predicts employee attrition risk using Machine Learning.

Model Used:
• Random Forest

Risk Categories:
🟢 Low
🟡 Medium
🔴 High
""")

dept = df.groupby("Department")["Attrition"].agg(
    Employees="count",
    Employees_Left="sum"
)

st.dataframe(dept)
