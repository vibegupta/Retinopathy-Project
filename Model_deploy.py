
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from colored import fg


import pickle
with open("N_model.sav","rb") as f:
    x=pickle.load(f)
with open("scaler.pkl","rb") as h:
    scaler=pickle.load(h)



st.title('Model Deployment:Neural Network')
st.sidebar.header("Please Enter your details below")

def user_ip():
    Age = st.sidebar.number_input("Insert age")
    Systolic_bp = st.sidebar.number_input("Insert Systolic BP")
    Diastolic_bp = st.sidebar.number_input("Insert Diastolic_bp")
    Cholesterol = st.sidebar.number_input("Insert Cholesterol")
    data={'Age':Age,'Systolic_bp':Systolic_bp,'Diastolic_bp':Diastolic_bp,'Cholesterol':Cholesterol}
    features = pd.DataFrame(data,index=[0])
    return features
df=user_ip()
st.subheader("Your Entered data:")
st.dataframe(df)
st.subheader("Entered Data Standardized for Prediction..")
numeric_column=['age', 'systolic_bp', 'diastolic_bp', 'cholesterol']
data=pd.DataFrame(scaler.transform(df), columns=numeric_column)
st.write(data)
if x.predict(data)==True:
    outcome="You have Retinopathy"
else:
    outcome="You don't have Retinopathy"

show=f"Based on Entered data there's 75% Probability that {outcome}"
final_text=f'<p style="font-family:sans-serif; color:Maroon; font-size: 20px;">{show}</p>'
#"""st.write(final_text)"""
st.markdown(final_text, unsafe_allow_html=True)

def add_bg_from_url(link):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({link});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url(link="https://wallpapercave.com/wp/wp2968489.jpg")