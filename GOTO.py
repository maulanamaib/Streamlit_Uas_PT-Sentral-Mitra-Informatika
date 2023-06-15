import streamlit as st
import joblib
import datetime
import time
import webbrowser
import matplotlib.pyplot as plt
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
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Prepocessing Data", "Modelling", "Implementasi"])
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
    run = colom[1].button("Check")
    Mm = joblib.load('model/MinMax.pkl')
    rd = joblib.load('model/ReDim.pkl')
    if run :
        if nomalisasi == "MinMax Scaler":
            st.write(Mm)
        elif nomalisasi == "Reduksi dimensi":
            st.write(rd)
        # st.write("ok") 

with tab3:
    with st.expander("Model"):
        model = st.radio("""Metode Regresi yang digunakan :""",("K-Neighbors Regressor", "Stacking Regressor", "Random Forest Regressor"))
    columns = st.columns((2, 0.6, 2))
    submit = columns[1].button("Submit")
    knn = joblib.load('model/KNN.pkl')
    bgr = joblib.load('model/LinearRegression.pkl')
    rf = joblib.load('model/RandomForest.pkl')
    if submit :
        if model == "K-Neighbors Regressor":
            st.success(str(knn))
        elif model == "Stacking Regressor":
            st.success(str(bgr))
        elif model == "Random Forest Regressor":
            st.success(str(rf))
        # st.write("ok")
with tab4:
    st.write("")
    st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Prediksi Saham GOTO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>", unsafe_allow_html=True)
    date = st.date_input("Masukkan Tanggal:",datetime.date(2023, 9, 1))
    st.write("Range input from: 2023-09-01 to 2029-12-30")
    columns = st.columns((2, 0.6, 2))
    cek = columns[1].button("submit")
    if cek :
        # mengambil date month dan year
            date = pd.to_datetime(date)
            date = date.strftime("%Y-%m-%d")
            # jika date < 2023-01 maka akan mengambil dari csv yang sudah ada
            if date <= '2029-06':
                data = pd.read_csv("data/prediksiluck.csv")                
                # get index dari date yang diinput di data
                
                index = data[data['Days'] == date].index.values.astype(int)[0]
                # index = data['Month'] == date

                # jika index + 10 > 339 maka mengambil data dari index sampai 339 - 10
                if index + 10 > 339:
                    datas = data.iloc[339-9:340]
                else:
                    datas = data.iloc[index:index+10]
               
                # # pisahkan month ke dalam list
                Days = datas['Days'].tolist()
                df = datas['RF'].tolist()
                # data_fix = {}
                # for i in range(len(label)):
                #     data_fix['Month'] = month
                #     data_fix[label[i]] = data_select[i]
                    # memasukkan bulan ke dalam data_fix
                
                # membuat plt untuk menampilkan hasil prediksi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.grid(True)
                # for i in range(len(datas)):
                ax.plot(Days, df)
                ax.set_xlabel('Days')
                ax.set_ylabel('Saham')
                ax.set_title('Prediksi Saham GOTO')
                ax.legend()
                st.pyplot(fig)

                # # menampilkan hasil prediksi data_fix pada st.table
                datas = pd.DataFrame(datas)
                st_date = datas.head(1)['Days'].values[0]
                en_date = datas.tail(1)['Days'].values[0]
                st.markdown(f"<h3 style='text-align: center; color: white; margin:0 ; padding:0;'>Hasil Prediksi Saham Goto Pada {st_date} - {en_date}</h3>", unsafe_allow_html=True)
                datas = datas.style.set_properties(**{'text-align': 'center'})
                datas = datas.set_table_styles([ dict(selector='th', props=[('text-align', 'center')] ) ])
                st.table(datas)
            else:
                st.warning("Tanggal yang anda masukkan melebihi batas , Masih dalam tahap pengembangan ❤️")

                
        # else:
        #     st.warning("Pilih Metode Regresi")
        # st.write("cek")
