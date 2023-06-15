# import libary 
import streamlit as st
import joblib
import dataset
import datetime
import time
import webbrowser
# import BaggingRegersion
# import RandomForest
# import knn
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
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/maulanamaib/streamlit_hepa.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a><a href="https://maulanamaib.github.io/datamining/intro.html" target="_blank"><button  style="border-radius: 12px;position: relative; top:50%;"><i style="color: orange" class="fa fa-book"></i> Jupyter</button></a></div>', unsafe_allow_html=True)
 

# colum = st.columns((0.1,10,1.5))
# url = 'https://github.com/maulanamaib/streamlit_hepa.git'

# if colum[1].button('GitHub'):
#     webbrowser.open_new_tab(url)

# link = 'https://maulanamaib.github.io/datamining/intro.html'

# if colum[2].button('Jupyter'):
    # webbrowser.open_new_tab(link)
# colum = st.columns((0.1,10,1.5))
# github= colum[1].button("check out this [link]()")
# jupyter = colum[2].button("")






# insialisasi web
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Processing Data", "Modelling", "Implementasi"])
with tab1:
    kolom = st.columns((1 , 1.5))   
#     url = 'https://github.com/maulanamaib/streamlit_hepa.git'

#     if kolom[1].button('GitHub'):
#         webbrowser.open_new_tab(url)

   
  
    # home = kolom[1].button('Home')
#     about = kolom[2].button('About')

   
#     kolom[4].button('Click Me!', 'https://maulanamaib.github.io/datamining/intro.html')

#     link = 'https://maulanamaib.github.io/datamining/intro.html'

#     if kolom[4].button('Jupyter'):
#         webbrowser.open_new_tab(link)

    # home page
#     if home==False and about==False or home==True and about==False:
        
    

    
  
          
           
    # about page
#     if about==True and home==False:
#         st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>", unsafe_allow_html=True)
#         st.write('Sistem Predeksi Penyakit hepa adalah sebuah sistem yang bertujuan untuk memprediksi penyakit hepa. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit.')
#         st.markdown("<p  color: white;'>Pada sistem ini menggunakan model KNN ( <i>K-nearest neighbors algorithm</i> ) dengan parameter <b>K = 3</b> . Dataset yang digunakan memiliki <b>5 fitur</b> termasuk kelas.</p>", unsafe_allow_html=True)
#         st.write('Alasan menggunakan model KNN dengan parameter k = 3 adalah karena memiliki akurasi yang terbesar dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.')
#         st.write("Disini range umur mempengaruhi prediksi kemungkinan terkenanya penyakit hepa")
#         st.markdown("<b>Alanine transminase (ALT), yaitu enzim yang mengubah protein menjadi energi untuk digunakan oleh sel-sel hati<b>",unsafe_allow_html=True)
#         st.markdown("<b>Alanine transminase (ALT), yaitu enzim yang mengubah protein menjadi energi untuk digunakan oleh sel-sel hati<b>",unsafe_allow_html=True)

with tab2:
    with st.expander("Normalizer"):
        nomalisasi = st.radio("Normalisasi yang digunakan",("MinMax Scaler", "Reduksi dimensi"))
    colom = st.columns((2, 0.6, 2))
    run = colom[1].button("run  ")
    Mm = joblib.load('MinMax.pkl')
    rd = joblib.load('ReDim.pkl')
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
    knn = joblib.load('knn.pkl')
    bgr = joblib.load('BaggingRegressor.pkl')
    rf = joblib.load('RandomForest.pkl')
    if submit :
        if model == "K-Neighbors Regressor":
            st.write(knn)
        elif model == "Bagging Regressor":
            st.write(bgr)
        elif model == "Random Forest Regressor":
            st.write(rf)
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