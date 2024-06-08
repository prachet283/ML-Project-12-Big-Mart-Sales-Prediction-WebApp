# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:27:35 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/model.pkl", 'rb') as f:
    model1 = pickle.load(f)

with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/model2.pkl", 'rb') as f:
    model2 = pickle.load(f)
    
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/model3.pkl", 'rb') as f:
    model3 = pickle.load(f)

with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-12-Big Mart Sales Prediction/encoders.pkl", 'rb') as f:
    encoders = pickle.load(f)
    

#creating a function for prediction

def big_mart_sales_prediction1(input_data):
    
    encoded_data = list(input_data)
    
    columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    for i, col in enumerate(columns):
        if col in encoders:
            encoded_data[i] = encoders[col].transform([encoded_data[i]])[0]
    
    encoded_data = np.array(encoded_data).reshape(1, -1)
    
    prediction = model1.predict(encoded_data)

    return prediction[0]

def big_mart_sales_prediction2(input_data):
    
    encoded_data = list(input_data)
    
    columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    for i, col in enumerate(columns):
        if col in encoders:
            encoded_data[i] = encoders[col].transform([encoded_data[i]])[0]
    
    encoded_data = np.array(encoded_data).reshape(1, -1)
    
    prediction = model2.predict(encoded_data)

    return prediction[0]

def big_mart_sales_prediction3(input_data):
    
    encoded_data = list(input_data)
    
    columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    for i, col in enumerate(columns):
        if col in encoders:
            encoded_data[i] = encoders[col].transform([encoded_data[i]])[0]
    
    encoded_data = np.array(encoded_data).reshape(1, -1)
    
    prediction = model3.predict(encoded_data)

    return prediction[0]




def main():
    
    #giving a title
    st.title('Big Mart Sales Prediction Web App')
    
    col1 , col2 = st.columns(2)
    #getting input data from user
    with col1:
        Item_Identifier = st.text_input("Item Identifier")
    with col2:
        Item_Weight = st.number_input("Item Weight")
    with col1:
        Item_Fat_Content = st.selectbox('Item Fat Content',('Low Fat', 'Regular'))
    with col2:
        Item_Visibility	 = st.number_input("Item Visibility")
    with col1:
        Item_Type = st.selectbox("Item Type",('Fruits and Vegetables','Snack Foods','Household','Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Soft Drinks','Meat','Breads','Hard Drinks','Starchy Foods','Breakfast','Seafood','Others'))
    with col2:
        Item_MRP = st.number_input("Item MRP")
    with col1:
        Outlet_Identifier = st.selectbox("Outlet_Identifier",('OUT010','OUT013','OUT017','OUT018','OUT019','OUT027','OUT035','OUT045','OUT046','OUT049'))
    with col2:
        Outlet_Establishment_Year = int(st.selectbox("Outlet Establishment Year",('1985','1987','1997','1998','1999','2002','2004','2007','2009')))
    with col1:
        Outlet_Size	= st.selectbox("Outlet_Size",('Small', 'Medium','High'))
    with col2:
        Outlet_Location_Type = st.selectbox("Outlet_Location_Type",('Tier 1', 'Tier 2','Tier 3'))
    with col1:
        Outlet_Type = st.selectbox("Outlet_Type",('Grocery Store', 'Supermarket Type1','Supermarket Type2','Supermarket Type3'))
    
    
    # code for prediction
    sales1 = ''
    sales2 = ''
    
    #creating a button for Prediction
    if st.button('Predict Big Mart Sales using Model1'):
        sales1 = big_mart_sales_prediction1((Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type))
        
    st.success('The Predicted Sales: '+ str(sales1)+'$'+('(XGBoost)'))
    
    
    if st.button('Predict Big Mart Sales using Model2'):
        sales2 = big_mart_sales_prediction2((Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type))
        
    st.success('The Predicted Sales: '+ str(sales2)+'$'+('(Random Forest)'))
    
    
    if st.button('Predict Big Mart Sales using Model3'):
        sales3 = big_mart_sales_prediction3((Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type))
        
    st.success('The Predicted Sales: '+ str(sales3)+'$'+('(Decision Tree)'))
    
    
    
    
if __name__ == '__main__':
    main()