#  Job Acceptance Prediction System (HR Analytics)

An end-to-end **Machine Learning & HR Analytics** solution designed to predict candidate job offer acceptance probability, identify key drivers of offer dropouts, and empower recruitment teams with actionable insights through an interactive **Streamlit** dashboard.



##  Problem Statement & Business Objectives
Recruitment and placement teams evaluate thousands of candidates across academic, technical, experiential, and compensation metrics, yet many eligible candidates reject final offers. Using a dataset of **51,500 candidate records** (50,000 unique records post-cleaning), this project addresses the following goals:
* **Predict Offer Acceptance:** Classify whether a candidate will accept (`Placed` = 1) or reject (`Not Placed` = 0) a job offer.
* **Identify Key Drivers:** Uncover which technical, experiential, and compensation factors most strongly influence acceptance decisions.
* **Reduce Offer Dropouts:** Flag high-risk candidates early so HR teams can optimize compensation offers, role alignment, and interview strategies.



##  Repository Structure
```text
Job-Acceptance-Prediction-System/
├── HR_Job_Placement_Dataset.csv      # Raw dataset (51,500 rows, 26 columns)
├── 01_cleaned_hr_data.csv            # Cleaned & deduplicated dataset (50,000 rows, 26 columns)
├── 02_featured_hr_data.csv           # Feature-engineered dataset (50,000 rows, 35 columns)
├── data_cleaning.ipynb               # Data auditing, string standardization, deduplication & imputation
├── eda_and_features.ipynb            # Exploratory Data Analysis (EDA), analyst tasks & feature engineering
├── ml_modeling.ipynb                 # Preprocessing pipeline, model comparison & feature importance
├── best_job_acceptance_model.pkl     # Serialized Gradient Boosting classification pipeline
├── app.py                            # Interactive Streamlit analytics dashboard & live predictor
└── README.md                         # Project documentation, insights & setup guide
```

---

##  End-to-End Methodology

### 1. Data Cleaning & Preprocessing (`data_cleaning.ipynb`)
* **String Standardization:** Stripped leading/trailing whitespace and normalized casing across categorical fields (`gender`, `company_tier`, `internship_experience`) prior to deduplication.
* **Deduplication:** Identified and removed **1,500 exact duplicate rows**, resulting in **50,000 unique candidate records**.
* **Domain-Aware Imputation:** Enforced logical consistency for freshers (`years_of_experience == 0` defaulted to `0` notice period/employment gap and `'No'` layoff history), followed by median imputation for remaining numerical nulls and mode imputation for categorical nulls.

### 2. Feature Engineering & EDA (`eda_and_features.ipynb`)
* **Derived Features:**
  * `experience_category`: Categorized into *Fresher* (0 yrs), *Junior* (1–2 yrs), and *Senior* (3+ yrs).
  * `academic_band`: Composite average of SSC, HSC, and Degree percentages (*Average*, *Good*, *Distinction*).
  * `skills_match_level`: Binned into *Low*, *Medium*, and *High*.
  * `avg_interview_score` & `interview_performance`: Composite metric combining Technical, Aptitude, and Communication scores.
  * `ctc_hike_ratio`: Relative salary expectation `(expected_ctc_lpa - previous_ctc_lpa) / previous_ctc_lpa`.
  * `high_risk_flag`: Flags candidates with >60% expected CTC hike, ≥60-day notice period, or job role mismatch.
* **HR Analyst Tasks Covered:**
  * Candidate Performance Analysis (*Academic bands vs placement outcome, skills match vs interview performance, certification impact*).
  * Placement & Acceptance Analysis (*Acceptance rate by company tier and experience level*).
  * Interview & Evaluation Analysis (*Technical, aptitude, communication, and overall interview scores vs placement probability*).

---

##  Key Analytical Insights
1. **Experience & Skills Match Drive Acceptance:** Offer acceptance increases sharply from **~14% for Freshers** to **~29% for Juniors** and **>50% for Seniors**. Candidates with a **High Skills Match** accept at **~39%** compared to **~11%** for Low Skills Match.
2. **Practical Skills Outweigh Academic Marks:** `technical_score` (**r = +0.40**) and `avg_interview_score` (**r = +0.34**) show the strongest positive correlation with acceptance, whereas past academic percentages (`ssc`, `hsc`, `degree`) show near-zero correlation (**r ≈ 0.00**).
3. **Compensation Expectations & Market Competition Cause Dropouts:** Higher `expected_ctc_lpa` (**r = -0.16**) and aggressive `ctc_hike_ratio` (**r = -0.15**) negatively impact acceptance, while high market competition drops overall acceptance from **~32% to ~24%**.
4. **Uniform Performance Across Company Tiers:** Acceptance rates remain consistent (~30%) across Tier 1, Tier 2, and Tier 3 organizations.

---

##  Machine Learning Results (`ml_modeling.ipynb`)
Three classification algorithms were trained and evaluated on a stratified 80/20 train-test split (10,000 test records) using a `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) pipeline:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Selected)** | **90.53%** | **0.8535** | **0.8295** | **0.8413** | **0.9671** |
| Random Forest | 87.49% | 0.8444 | 0.7191 | 0.7767 | 0.9493 |
| Logistic Regression | 86.90% | 0.7990 | 0.7578 | 0.7778 | 0.9394 |

**Top Predictive Features:** `technical_score`, `years_of_experience`, `skills_match_percentage`, `job_role_match`, `ctc_hike_ratio`, and `avg_interview_score`.

---

##  Streamlit Dashboard (`app.py`)
The interactive web application provides:
* **7 Executive KPIs:** Total Candidates, Placement Rate (%), Job Acceptance Rate (%), Offer Dropout Rate (%), Average Interview Score, Average Skills Match %, and High-Risk Candidate Percentage.
* **Dynamic Filtering:** Slice recruitment metrics by Company Tier, Experience Category, and Competition Level.
* **Live Offer Acceptance Predictor:** Real-time probability scoring and risk classification for individual candidate profiles.



