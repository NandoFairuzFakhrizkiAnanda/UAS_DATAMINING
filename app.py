import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('LogisticRegression_model.pkl','rb'))

#load dataset
data = pd.read_csv('Breast_Cancer.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Kanker Payudara')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['SVM','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Breast Cancer</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('fractal_dimension_worst',axis=1)
y = data['fractal_dimension_worst']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    radius = st.sidebar.slider('radius_mean',0,20,1)
    texture = st.sidebar.slider('texture_mean',0,200,108)
    perimeter = st.sidebar.slider('perimeter_mean',0,140,40)
    area = st.sidebar.slider('area_mean',0,100,25)
    smoothness = st.sidebar.slider('smoothness_mean',0,1000,120)
    compactness = st.sidebar.slider('compactness_mean',0,80,25)
    concavity = st.sidebar.slider('concavity_mean', 0.05,2.5,0.45)
    concave = st.sidebar.slider('concave points_mean',21,100,24)
    symmetry = st.sidebar.slider('symmetry_mean',21,100,24)
    radiusWorst = st.sidebar.slider('radius_worst',0,20,1)
    textureWorst = st.sidebar.slider('texture_worst',0,200,108)
    perimeterWorst = st.sidebar.slider('perimeter_worst',0,140,40)
    areaWorst = st.sidebar.slider('area_worst',0,100,25)
    smoothnessWorst = st.sidebar.slider('smoothness_worst',0,1000,120)
    compactnessWorst = st.sidebar.slider('compactness_worst',0,80,25)
    concavityWorst = st.sidebar.slider('concavity_worst', 0.05,2.5,0.45)
    concavePointsWorst = st.sidebar.slider('concave points_worst',21,100,24)
    symmetryWorst = st.sidebar.slider('symmetry_worst',21,100,24)

    user_report_data = {
        'radius_mean':radius,
        'texture_mean':texture,
        'perimeter_mean':perimeter,
        'area_mean':area,
        'smoothness_mean':smoothness,
        'compactness_mean':compactness,
        'concavity_mean':concavity,
        'concave points_mean':concave,
        'symmetry_mean':symmetry,
        'radius_worst':radiusWorst,
        'texture_worst':textureWorst,
        'perimeter_worst':perimeterWorst,
        'area_worst':areaWorst,
        'smoothness_worst':smoothnessWorst,
        'compactness_worst':compactnessWorst,
        'concavity_worst':concavityWorst,
        'concave points_worst':concavePointsWorst,
        'symmetry_worst':symmetryWorst,

    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena Kanker Payudara'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')



