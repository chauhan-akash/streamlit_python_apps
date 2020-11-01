#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
# 1. import key libraries
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import precision_score, recall_score


############################################################
# 2. main function - defining basic app headers & load data
############################################################

def main():
    st.title("Binary classification web app")
    st.sidebar.title("Interactive controls")
    st.markdown("Are your mushrooms poisnous?")
    
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("./data/mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df):
        y = df['type']
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1234)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            cm = plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot(cm.figure_)
            
        if "ROC Curve" in metrics_list:
            st.subheader("ROC curve")
            roc = plot_roc_curve(model, x_test, y_test)
            st.pyplot(roc.figure_)
            
        if "Precision Recall curve" in metrics_list:
            st.subheader("Precision Recall curve")
            pr = plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot(pr.figure_)
    
    df = load_data()
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushrooms dataset used for classification - ")
        st.write(df)
    
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisnous']
    
    st.sidebar.subheader("1. Choose Classifier")
    class_options = ('Support Vector Machines (SVM)', 'Logistic Regression', 'Random Forest')
    classifier = st.sidebar.selectbox("List of Classifiers", class_options)
    
    if classifier == 'Support Vector Machines (SVM)':
        st.sidebar.subheader("2. Model Hyperparameters")
        c = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='c')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)", ("scale", "auto"))
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machines (SVM) results")
            model = SVC(C=c, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", np.round(accuracy,2))
            st.write("Precision : ", np.round(precision_score(y_test, y_pred, labels=class_names),2))
            st.write("Recall : ", np.round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)
        
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("2. Model Hyperparameters")
        c = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='c_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations - ", 100, 500,key="max_it")
        
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression results")
            model = LogisticRegression(C=c,max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", np.round(accuracy,2))
            st.write("Precision : ", np.round(precision_score(y_test, y_pred, labels=class_names),2))
            st.write("Recall : ", np.round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)
            
            
    if classifier == 'Random Forest':
        st.sidebar.subheader("2. Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Enter the number of trees in random forest - ", 100, 2000 , step=10, key="n_est")
        max_depth = st.sidebar.number_input("Maximum depth of trees in random forest - ", 1, 20 , step=1, key="max_dep")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees?", ('True', 'False'), key='bootstrap')
        
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall curve"))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", np.round(accuracy,2))
            st.write("Precision : ", np.round(precision_score(y_test, y_pred, labels=class_names),2))
            st.write("Recall : ", np.round(recall_score(y_test, y_pred, labels=class_names),2))
            plot_metrics(metrics)
        

if __name__ == '__main__':
    main()



