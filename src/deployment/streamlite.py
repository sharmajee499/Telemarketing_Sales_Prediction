import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import lime.lime_tabular


# ******************************************************************************
# INPUT BUTTONS AND TEXTBOX
# ******************************************************************************

st.header("Telemarketing Success Prediction")

st.sidebar.title("Please, enter the details of the respective customer")

# Age Input
input_age = st.sidebar.number_input("Customer's Age", min_value=18, max_value=140)

# Job Input
job_list = [
    "housemaid",
    "services",
    "admin.",
    "blue-collar",
    "technician",
    "retired",
    "management",
    "unemployed",
    "self-employed",
    "entrepreneur",
    "student",
]
input_job = st.sidebar.selectbox("Customer's Job", options=job_list)

# Maritial Status
marital_list = ["married", "single", "divorced"]
input_marital = st.sidebar.selectbox("Marital Status", options=marital_list)

# Education Status
education_list = [
    "basic.4y",
    "high.school",
    "basic.6y",
    "basic.9y",
    "professional.course",
    "university.degree",
    "illiterate",
]
input_education = st.sidebar.selectbox("Education Level", options=education_list)

# Housing Loan
input_housing = st.sidebar.selectbox("Housing Loan", options=["yes", "no"])

# Personal Loan
input_personal_loan = st.sidebar.selectbox("Personal Loan", options=["yes", "no"])

# Contact
contact_list = ["telephone", "cellular"]
input_contact = st.sidebar.selectbox("Customer Contacted With", options=contact_list)

# Month
month_list = ["may", "jun", "jul", "aug", "oct", "nov", "dec", "mar", "apr", "sep"]
input_month = st.sidebar.selectbox("Month Contacted", month_list)

# Day of Week
day_list = ["mon", "tue", "wed", "thu", "fri"]
input_day = st.sidebar.selectbox("Cantact Day", options=day_list)

# Campaign
input_campaign = st.sidebar.number_input(
    "# of Contact Performed this campaign", min_value=0
)

# Contacted previous campaign
input_previous_campaign = st.sidebar.number_input(
    "# of Contacts On Previous Campaign", min_value=0
)

# Previous Outcome
poutcome_list = ["nonexistent", "failure", "success"]
input_poutcome = st.sidebar.selectbox("Previous Campaign Result", options=poutcome_list)

# Employment Variation Rate
input_emp_var_rate = st.sidebar.number_input(label="Employee Variation Rate")

# Consumer Price Index
input_cons_price_ind = st.sidebar.number_input(label="Consumer Price Index")

# Consumer Confidence Index
input_cons_confd_ind = st.sidebar.number_input("Consumer Confidence Index")

# Eurobar 3-month Rate
input_eurobar = st.sidebar.number_input("Eurobar (3 Months) Rate")

# Number of Employees
input_employees = st.sidebar.number_input(
    "Quarterly Average of Total number of Employeed Citizen"
)


# ******************************************************************************
# Formating the Input
# ******************************************************************************
feature_names = [
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

# list of all provided input
all_input = [
    input_age,
    input_job,
    input_marital,
    input_education,
    input_housing,
    input_personal_loan,
    input_contact,
    input_month,
    input_day,
    input_campaign,
    input_previous_campaign,
    input_poutcome,
    input_emp_var_rate,
    input_cons_price_ind,
    input_cons_confd_ind,
    input_eurobar,
    input_employees,
]

# Loading the train data
X = np.load(
    'src/deployment/X.npy',
    allow_pickle=True,
)

# Copy of X
X_lime = np.copy(X)

# Transformation
cat_features_ind = [1, 2, 3, 4, 5, 6, 7, 8, 11]

# Categorical Name
categorical_names = {}

for ind in cat_features_ind:
    le = LabelEncoder()
    X[:, ind] = le.fit_transform((X[:, ind]))
    categorical_names[ind] = le.classes_
    all_input[ind] = le.transform([all_input[ind]])

input_arr = []
for elements in all_input:
    try:
        input_arr.append(int(elements[0]))
    except:
        input_arr.append(int(elements))

encoder = ColumnTransformer(
    [("ohe", OneHotEncoder(), cat_features_ind)], remainder="passthrough"
)

data = encoder.fit(X)

# ******************************************************************************
# Prediction
# ******************************************************************************

# Open the pickled model
with open(
    "src/deployment/lgbm_final_model",
    "rb",
) as f:
    lgbm_model = pickle.load(f)

# transforming the 'input_arr'
input_encoded = data.transform(np.array(input_arr).reshape(1, -1))

# Prediction
predict = lgbm_model.predict_proba(encoder.transform(input_encoded))
predict2 = lgbm_model.predict(encoder.transform(input_encoded))

# ******************************************************************************
# Explaination
# ******************************************************************************

# Predict function for the LIME
predict_fn = lambda x: lgbm_model.predict_proba(encoder.transform(x))

# LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X,
    feature_names=feature_names,
    class_names=["No", "Yes"],
    discretize_continuous=True,
    categorical_features=cat_features_ind,
    categorical_names=categorical_names,
    kernel_width=3,
)

# Instance Explaination
np.random.seed(112233)
exp = explainer.explain_instance(np.array(all_input), predict_fn, num_features=7)


col1, col2, col3 = st.columns(3)
prob_zero = exp.predict_proba[0] * 100
prob_one = exp.predict_proba[1] * 100
col1.metric("Buying Probability %", f"{prob_one:.2f}")
col2.metric("Rejecting Probability %", f"{prob_zero:.2f}")
col3.metric(f"Threshold %", value="19")

st.caption(
    """Since our threshold is 19%, if the buying probability is greater than 19%, 
    that means customer will buy the deposite."""
)


# Plot
st.pyplot(exp.as_pyplot_figure())
