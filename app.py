import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("churn_model.pkl")

# -----------------------
# HEADER
# -----------------------
st.markdown("<h1 style='text-align: center;'>📊 Telecom Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# SIDEBAR INPUTS
# -----------------------
st.sidebar.header("🧾 Customer Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)

region_circle = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
connection_type = st.sidebar.selectbox("Connection Type", ["Prepaid", "Postpaid"])
plan_type = st.sidebar.selectbox("Plan Type", ["Basic", "Premium", "Gold"])

contract_type = st.sidebar.selectbox("Contract Type", ["Monthly", "Yearly"])
base_plan_category = st.sidebar.selectbox("Base Plan", ["Low", "Medium", "High"])

tenure_months = st.sidebar.slider("Tenure (months)", 0, 120, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=500.0)
total_charges = st.sidebar.number_input("Total Charges", value=6000.0)

avg_data_gb_month = st.sidebar.number_input("Data Usage (GB)", value=10.0)
avg_voice_mins_month = st.sidebar.number_input("Voice Minutes", value=300.0)
sms_count_month = st.sidebar.number_input("SMS Count", value=50)

overage_charges = st.sidebar.number_input("Overage Charges", value=0.0)

is_family_plan = st.sidebar.selectbox("Family Plan", [0,1])
is_multi_service = st.sidebar.selectbox("Multi Service", [0,1])

network_issues_3m = st.sidebar.number_input("Network Issues", value=2)
dropped_call_rate = st.sidebar.slider("Dropped Call Rate", 0.0, 1.0, 0.05)

avg_data_speed_mbps = st.sidebar.number_input("Speed (Mbps)", value=20.0)

num_complaints_3m = st.sidebar.number_input("Complaints (3m)", value=1)
num_complaints_12m = st.sidebar.number_input("Complaints (12m)", value=5)

call_center_interactions_3m = st.sidebar.number_input("Call Center Calls", value=3)

last_complaint_resolution_days = st.sidebar.number_input("Resolution Days", value=5)

app_logins_30d = st.sidebar.number_input("App Logins", value=10)
selfcare_transactions_30d = st.sidebar.number_input("Selfcare Transactions", value=5)

auto_pay_enrolled = st.sidebar.selectbox("Auto Pay", [0,1])
late_payment_flag_3m = st.sidebar.selectbox("Late Payment", [0,1])

avg_payment_delay_days = st.sidebar.number_input("Payment Delay", value=2)

arpu = st.sidebar.number_input("ARPU", value=500.0)

segment_value = st.sidebar.selectbox("Segment", ["Low", "Medium", "High"])

nps_score = st.sidebar.slider("NPS Score", -100, 100, 20)
service_rating_last_6m = st.sidebar.slider("Service Rating", 1, 5, 3)

received_competitor_offer_flag = st.sidebar.selectbox("Competitor Offer", [0,1])
retention_offer_accepted_flag = st.sidebar.selectbox("Retention Accepted", [0,1])

# Feature engineering
avg_revenue_per_month = total_charges / (tenure_months + 1)

# -----------------------
# INPUT DATA
# -----------------------
input_data = pd.DataFrame([{
    'gender': gender,
    'age': age,
    'region_circle': region_circle,
    'connection_type': connection_type,
    'plan_type': plan_type,
    'contract_type': contract_type,
    'base_plan_category': base_plan_category,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'avg_data_gb_month': avg_data_gb_month,
    'avg_voice_mins_month': avg_voice_mins_month,
    'sms_count_month': sms_count_month,
    'overage_charges': overage_charges,
    'is_family_plan': is_family_plan,
    'is_multi_service': is_multi_service,
    'network_issues_3m': network_issues_3m,
    'dropped_call_rate': dropped_call_rate,
    'avg_data_speed_mbps': avg_data_speed_mbps,
    'num_complaints_3m': num_complaints_3m,
    'num_complaints_12m': num_complaints_12m,
    'call_center_interactions_3m': call_center_interactions_3m,
    'last_complaint_resolution_days': last_complaint_resolution_days,
    'app_logins_30d': app_logins_30d,
    'selfcare_transactions_30d': selfcare_transactions_30d,
    'auto_pay_enrolled': auto_pay_enrolled,
    'late_payment_flag_3m': late_payment_flag_3m,
    'avg_payment_delay_days': avg_payment_delay_days,
    'arpu': arpu,
    'segment_value': segment_value,
    'nps_score': nps_score,
    'service_rating_last_6m': service_rating_last_6m,
    'received_competitor_offer_flag': received_competitor_offer_flag,
    'retention_offer_accepted_flag': retention_offer_accepted_flag,
    'avg_revenue_per_month': avg_revenue_per_month
}])

# -----------------------
# PREDICTION
# -----------------------
st.markdown("### 🔍 Prediction")

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", "Churn" if prediction==1 else "No Churn")
    col2.metric("Probability", f"{probability:.2%}")

    risk = "🟢 Low" if probability < 0.3 else "🟡 Medium" if probability < 0.7 else "🔴 High"
    col3.metric("Risk Level", risk)

    st.markdown("---")

    st.subheader("📊 Churn Probability")
    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    # -----------------------
    # SHAP EXPLANATION
    # -----------------------
    st.markdown("## 🧠 Why this prediction?")

    X_transformed = model.named_steps['preprocessor'].transform(input_data)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    explainer = shap.Explainer(model.named_steps['classifier'])
    shap_values = explainer(X_transformed)

    # ✅ FIXED FOR CLASSIFICATION
    shap_impact = shap_values.values[0, :, 1]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_impact
    })

    # Clean feature names (remove prefixes)
    shap_df["feature"] = shap_df["feature"].str.replace("num__", "")
    shap_df["feature"] = shap_df["feature"].str.replace("cat__", "")

    shap_df["abs_impact"] = shap_df["impact"].abs()
    top_shap = shap_df.sort_values("abs_impact", ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(top_shap["feature"], top_shap["impact"])
    ax.invert_yaxis()
    ax.set_title("Top Factors Affecting This Prediction")
    st.pyplot(fig, use_container_width=True)

    # -----------------------
    # TOP REASONS TEXT
    # -----------------------
    st.markdown("## 📌 Top Reasons")

    for _, row in top_shap.iterrows():
        if row["impact"] > 0:
            st.write(f"🔴 {row['feature']} increases churn risk")
        else:
            st.write(f"🔵 {row['feature']} decreases churn risk")

    # -----------------------
    # SMART SUMMARY
    # -----------------------
    st.markdown("## 🧠 Summary")

    if prediction == 1:
        st.warning("Customer is likely to churn mainly due to the above risk factors.")
    else:
        st.success("Customer is likely to stay due to strong engagement and positive indicators.")

    # -----------------------
    # PDF DOWNLOAD
    # -----------------------
    st.markdown("## 📄 Download Report")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Telecom Churn Prediction Report", styles["Title"]))
    content.append(Paragraph(f"Prediction: {'Churn' if prediction==1 else 'No Churn'}", styles["Normal"]))
    content.append(Paragraph(f"Probability: {probability:.2%}", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)

    st.download_button(
        label="Download PDF Report",
        data=buffer,
        file_name="churn_report.pdf",
        mime="application/pdf"
    )
