import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('diabetes.csv')

diabetes_mean_df = data.groupby('Outcome').mean()

X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

scalr = StandardScaler()
scalr.fit(X)
X = scalr.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

def app():
    img = Image.open(r'img.jpeg')
    img = img.resize((200,200))
    st.image(img, caption="Diabetes IMG", width=200)
    
    st.title("Diabetes Disease Prediction")
    st.sidebar.title("Input feature")
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    input_data = (preg, glucose, bp, skinthickness, insulin, bmi, dpf, age)
    np_arry_data = np.asarray(input_data)
    reshaped_data = np_arry_data.reshape(1,-1)
    
    prediction = model.predict(reshaped_data)
    
    if prediction[0]==1:
        st.warning("This person has diabetes!")
    else:
        st.success("This person do not have diabetes!")
    

if __name__ == '__main__':
    app()