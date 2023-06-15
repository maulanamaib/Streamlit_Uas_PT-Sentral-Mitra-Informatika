import streamlit as st
import joblib
import datetime
import time
import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components
# pige title
st.set_page_config(
    page_title="",
    page_icon="https://cdn-icons-png.flaticon.com/128/254/254207.png",
)

    # 0 = tidak ada penyakit hepa
    # 1 = ada penyakit hepa

# hide menu
hide_streamlit_style = """



<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

"""


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/maulanamaib/Streamlit_Uas_PT-Sentral-Mitra-Informatika.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a></div>', unsafe_allow_html=True)
 
# insialisasi web
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Processing Data", "Modelling", "Implementasi"])
with tab1:
    kolom = st.columns((1 , 1.5))   
    st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>", unsafe_allow_html=True)
    st.write('Sistem ini dibuat untuk membantu memprediksi saham dari perusahaan GOTO dengan menggunakan bahasa python dan frame web streamlit.')
    st.markdown("<p  color: white;'>Pada sistem ini menggunakan model Random Forest Regression ( <i>Random Forest Regression algorithm</i> ) dengan parameter <b>max_depth=d</b> . Dataset yang digunakan memiliki <b>6 fitur</b> termasuk Volume yang digunakan untuk fitur prediksi.</p>", unsafe_allow_html=True)
    st.write('Alasan menggunakan model Random Forest Regression dengan parameter <b>max_depth=d</b> adalah karena memiliki akurasi yang terbaik dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.')
    st.write("Disini range tahun mempengaruhi prediksi saham ke depannya tergantuung parameter yang digunakan.")
    st.write("Data ini diperoleh dari https://finance.yahoo.com/quote/LUCK.JK/history?p=LUCK.JK.")
    st.markdown("<b>Volume, menunjukkan jumlah saham atau kontrak yang diperdagangkan pada tanggal tertentu<b>",unsafe_allow_html=True)
    st.markdown("<b>Date, menunjukkan tanggal tertentu<b>",unsafe_allow_html=True)

with tab2:
    with st.expander("Normalizer"):
        nomalisasi = st.radio("Normalisasi yang digunakan",("MinMax Scaler", "Reduksi dimensi"))
    colom = st.columns((2, 0.6, 2))
    run = colom[1].button("run")
    Mm = joblib.load('model/MinMax.pkl')
    rd = joblib.load('model/ReDim.pkl')
    if run :
        if nomalisasi == "MinMax Scaler":
            st.write(Mm)
        elif nomalisasi == "Reduksi dimensi":
            st.write(rd)
        # st.write("ok") 

with tab3:
    with st.expander("Setting"):
        model = st.radio("""Metode Regresi yang digunakan :""",("K-Neighbors Regressor", "Bagging Regressor", "Random Forest Regressor"))
    columns = st.columns((2, 0.6, 2))
    submit = columns[1].button("Submit")
    knn = joblib.load('model/KNN.pkl')
    bgr = joblib.load('model/BaggingRegressor.pkl')
    rf = joblib.load('model/RandomForest.pkl')
    if submit :
        if model == "K-Neighbors Regressor":
            st.write("{knn}")
        elif model == "Stacking Regressor":
            st.write("{bgr}")
        elif model == "Random Forest Regressor":
            st.write("{rf}")
        # st.write("ok")
with tab4:
    st.write("")
    st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Prediksi Saham GOTO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>", unsafe_allow_html=True)
    date = st.date_input("Masukkan Tanggal:",datetime.date(2023, 1, 1))
    columns = st.columns((2, 0.6, 2))
    cek = columns[1].button("submit")
    if cek :
        st.write("cek")
