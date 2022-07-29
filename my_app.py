import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("Churn Prediction")
st.text('Left or Stay?')

#image
img = Image.open("churns.jpg")
st.image(img, width=500)

#sidebar hearder
st.sidebar.header('Left or Stay Predictor')

# Departments
Departments=st.sidebar.selectbox("Departments ", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])

# Salary
salary=st.sidebar.selectbox("salary", ["low", "medium", "high"])

# Satisfaction Level
satisfaction_level=st.sidebar.number_input("satisfaction_level", min_value=0.00, max_value=1.00, step=0.01)

#Last Evaluation
last_evaluation = st.sidebar.number_input("last_evaluation:",min_value=0.00, max_value=1.00, step=0.01)

#average_monthly_hours
average_montly_hours=st.sidebar.number_input("average_monthly_hours:",min_value=0, max_value=500, step=1)

#number_project
number_project=st.sidebar.number_input("number_project:",min_value=0, max_value=25, step=1)

#time_spend_company
time_spend_company=st.sidebar.number_input("time_spend_company:",min_value=0, max_value=25, step=1)

#Work_accident
work_accident = st.sidebar.number_input("work accident:", min_value=0, max_value=1, step=1)

#promotion_last_5years
promotion_last_5years = st.sidebar.number_input("promotion_last_5years:", min_value=0, max_value=1, step=1)

import pickle
filename = 'gradient_boosting_model'
model = pickle.load(open(filename, 'rb'))


my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation":last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    "salary": salary,
    "Departments ": Departments
}

my_dict=pd.DataFrame.from_dict([my_dict])

from sklearn.preprocessing import OrdinalEncoder
scale_mapper = {"Low":0, "Medium":1, "High":2}
my_dict["salary"] = my_dict["salary"].replace(scale_mapper)
enc = OrdinalEncoder()
my_dict[["salary"]] = enc.fit_transform(my_dict[["salary"]])

from sklearn.preprocessing import OneHotEncoder
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
cat=pd.DataFrame(categorical_transformer.fit_transform(my_dict[['Departments ']]).toarray())
my_dict=my_dict.join(cat)
my_dict.drop('Departments ', axis=1, inplace=True)

columns_name=['satisfaction_level','last_evaluation',
              'number_project',  'average_montly_hours',
          'time_spend_company',         'Work_accident', 'promotion_last_5years',
                      'salary',                       0,
                             1,                       2,
                             3,                       4,
                             5,                       6,
                             7,                       8,
                             9]

my_dict = my_dict.reindex(columns=columns_name, fill_value=0)


if st.sidebar.button("Check"):
    pred = model.predict(my_dict)
    if pred==0:
            st.success("Employee will stay")
    else:
            st.success("Employee will left")
st.sidebar.info("Please fill all required fields..")