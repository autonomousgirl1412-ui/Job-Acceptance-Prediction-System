import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib


st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv('featured_hr_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('job_acceptance_model.pkl')

df = load_data()
model = load_model()

st.sidebar.header("🔍 Filter Candidate Data")
selected_tier = st.sidebar.multiselect(
    "Company Tier",
    options=sorted(df['company_tier'].unique()),
    default=sorted(df['company_tier'].unique())
)
selected_exp = st.sidebar.multiselect(
    "Experience Category",
    options=['Fresher', 'Junior', 'Senior'],
    default=['Fresher', 'Junior', 'Senior']
)
selected_comp = st.sidebar.multiselect(
    "Competition Level",
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

filtered_df = df[
    (df['company_tier'].isin(selected_tier)) &
    (df['experience_category'].isin(selected_exp)) &
    (df['competition_level'].isin(selected_comp))
]


st.title("🎯 HR Analytics: Job Acceptance Prediction System")
st.markdown("Analyze recruitment patterns, monitor offer dropout risks, and predict candidate acceptance probability using **Gradient Boosting (90.5% Accuracy | 0.967 ROC-AUC)**.")

st.subheader("📊 Key Recruitment & Acceptance KPIs")

total_candidates = len(filtered_df)

placement_rate = filtered_df['target_accepted'].mean() * 100 if total_candidates > 0 else 0
job_acceptance_rate = placement_rate
offer_dropout_rate = 100 - job_acceptance_rate if total_candidates > 0 else 0
avg_interview = filtered_df['avg_interview_score'].mean() if total_candidates > 0 else 0
avg_skills_match = filtered_df['skills_match_percentage'].mean() if total_candidates > 0 else 0
high_risk_pct = (filtered_df['high_risk_flag'] == 'High Risk').mean() * 100 if total_candidates > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Candidates", f"{total_candidates:,}")
kpi2.metric("Placement Rate (%)", f"{placement_rate:.2f}%")
kpi3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}%")
kpi4.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}%")

kpi5, kpi6, kpi7 = st.columns(3)
kpi5.metric("Avg Interview Score", f"{avg_interview:.1f} / 100")
kpi6.metric("Avg Skills Match %", f"{avg_skills_match:.1f}%")
kpi7.metric("High-Risk Candidate %", f"{high_risk_pct:.2f}%")

st.divider()


tab1, tab2 = st.tabs(["📈 Recruitment & Acceptance Analytics", "🤖 Live Offer Acceptance Predictor"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        exp_chart = filtered_df.groupby('experience_category', as_index=False)['target_accepted'].mean()
        exp_chart['Acceptance Rate (%)'] = exp_chart['target_accepted'] * 100
        fig1 = px.bar(
            exp_chart, x='experience_category', y='Acceptance Rate (%)',
            color='experience_category', title="Acceptance Rate by Experience Category",
            text_auto='.1f'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        skills_chart = filtered_df.groupby('skills_match_level', as_index=False)['target_accepted'].mean()
        skills_chart['Acceptance Rate (%)'] = skills_chart['target_accepted'] * 100
        fig2 = px.bar(
            skills_chart, x='skills_match_level', y='Acceptance Rate (%)',
            color='skills_match_level', title="Acceptance Rate by Skills Match Level",
            text_auto='.1f'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    
    with col3:
        tier_comp = filtered_df.groupby(['company_tier', 'competition_level'], as_index=False)['target_accepted'].mean()
        tier_comp['Acceptance Rate (%)'] = tier_comp['target_accepted'] * 100
        fig3 = px.bar(
            tier_comp, x='company_tier', y='Acceptance Rate (%)', color='competition_level',
            barmode='group', title="Acceptance Rate by Company Tier & Competition Level"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            filtered_df, x='status', y='technical_score', color='status',
            title="Technical Score Distribution by Placement Status"
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.subheader("Predict Candidate Job Offer Acceptance Probability")
    st.markdown("Adjust candidate attributes below to simulate acceptance likelihood and dropout risk.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age_years = st.slider("Age (Years)", 21, 35, 26)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ssc_percentage = st.slider("SSC Percentage", 50.0, 95.0, 70.0)
            hsc_percentage = st.slider("HSC Percentage", 50.0, 95.0, 72.0)
            degree_percentage = st.slider("Degree Percentage", 55.0, 95.0, 74.0)
            degree_specialization = st.selectbox(
                "Degree Specialization",
                ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Others"]
            )
            certifications_count = st.slider("Certifications Count", 0, 8, 2)
            internship_experience = st.selectbox("Internship Experience", ["Yes", "No"])

        with c2:
            technical_score = st.slider("Technical Score", 40.0, 100.0, 75.0)
            aptitude_score = st.slider("Aptitude Score", 40.0, 100.0, 68.0)
            communication_score = st.slider("Communication Score", 45.0, 100.0, 70.0)
            skills_match_percentage = st.slider("Skills Match Percentage", 45.0, 100.0, 80.0)
            years_of_experience = st.slider("Years of Experience", 0, 5, 2)
            relevant_experience = st.selectbox("Relevant Experience", ["Relevant", "Not Relevant"])
            job_role_match = st.selectbox("Job Role Match", ["Matched", "Not Matched"])
            employment_gap_months = st.slider("Employment Gap (Months)", 0, 18, 0)

        with c3:
            previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 1.0, 15.0, 5.0, step=0.5)
            expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 3.0, 27.0, 7.0, step=0.5)
            company_tier = st.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])
            competition_level = st.selectbox("Competition Level", ["Low", "Medium", "High"])
            bond_requirement = st.selectbox("Bond Requirement", ["Not Required", "Required"])
            notice_period_days = st.selectbox("Notice Period (Days)", [0, 15, 30, 45, 60, 90])
            layoff_history = st.selectbox("Layoff History", ["No", "Yes"])
            career_switch_willingness = st.selectbox("Career Switch Willingness", ["Not Willing", "Willing"])
            relocation_willingness = st.selectbox("Relocation Willingness", ["Willing", "Not Willing"])

        submitted = st.form_submit_button("🔮 Predict Acceptance Probability")

    if submitted:
        
        exp_cat = 'Fresher' if years_of_experience == 0 else ('Junior' if years_of_experience <= 2 else 'Senior')
        avg_acad = np.mean([ssc_percentage, hsc_percentage, degree_percentage])
        acad_band = 'Average' if avg_acad <= 65 else ('Good' if avg_acad <= 75 else 'Distinction')
        skills_lvl = 'Low' if skills_match_percentage <= 60 else ('Medium' if skills_match_percentage <= 80 else 'High')
        avg_int = np.mean([technical_score, aptitude_score, communication_score])
        int_perf = 'Needs Improvement' if avg_int <= 60 else ('Moderate' if avg_int <= 75 else 'Strong')
        ctc_hike = (expected_ctc_lpa - previous_ctc_lpa) / previous_ctc_lpa

        input_data = pd.DataFrame([{
            'age_years': age_years,
            'gender': gender,
            'ssc_percentage': ssc_percentage,
            'hsc_percentage': hsc_percentage,
            'degree_percentage': degree_percentage,
            'degree_specialization': degree_specialization,
            'technical_score': technical_score,
            'aptitude_score': aptitude_score,
            'communication_score': communication_score,
            'skills_match_percentage': skills_match_percentage,
            'certifications_count': certifications_count,
            'internship_experience': internship_experience,
            'years_of_experience': years_of_experience,
            'career_switch_willingness': career_switch_willingness,
            'relevant_experience': relevant_experience,
            'previous_ctc_lpa': previous_ctc_lpa,
            'expected_ctc_lpa': expected_ctc_lpa,
            'company_tier': company_tier,
            'job_role_match': job_role_match,
            'competition_level': competition_level,
            'bond_requirement': bond_requirement,
            'notice_period_days': notice_period_days,
            'layoff_history': layoff_history,
            'employment_gap_months': employment_gap_months,
            'relocation_willingness': relocation_willingness,
            'experience_category': exp_cat,
            'avg_academic_score': avg_acad,
            'academic_band': acad_band,
            'skills_match_level': skills_lvl,
            'avg_interview_score': avg_int,
            'interview_performance': int_perf,
            'ctc_hike_ratio': ctc_hike
        }])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown("### Prediction Result")
        res1, res2 = st.columns(2)
