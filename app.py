import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================
# KONFIGURASI
# ==========================

st.set_page_config(
    page_title="Prediksi Mahasiswa Dropout",
    page_icon="🎓",
    layout="wide"
)


# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data.csv", sep=";")

model = joblib.load("model_dropout.pkl")

scaler = joblib.load("scaler.pkl")

# ==========================
# SIDEBAR
# ==========================



st.sidebar.title("Prediksi Mahasiswa Dropout")

menu = st.sidebar.radio(
    "Pilih Menu",
    (
        "🏠 Home",
        "📊 Dashboard EDA",
        "🤖 Model Demo",
        "📈 Evaluasi Model",
        "💡 Interpretasi Hasil",
        "📖 Dokumentasi"
    )
)

# ===========================================
# HOME
# ===========================================

if menu == "🏠 Home":

    st.title("🎓 Sistem Prediksi Mahasiswa Dropout")

    st.markdown("""
## Selamat Datang

Aplikasi ini dibuat untuk memprediksi kemungkinan mahasiswa mengalami **Dropout**
menggunakan algoritma **Random Forest Classifier**.

### Dataset
Predict Students Dropout and Academic Success

### Algoritma
Random Forest

### Fitur

- Dashboard EDA
- Model Demo
- Evaluasi Model
- Interpretasi Hasil
- Dokumentasi

Silakan pilih menu pada sidebar.
""")

   
# ===========================================
# DASHBOARD EDA
# ===========================================

elif menu == "📊 Dashboard EDA":

    st.title("📊 Dashboard Exploratory Data Analysis (EDA)")

    st.write("Berikut merupakan hasil eksplorasi dataset Students Dropout and Academic Success.")

    st.divider()

    # ===============================
    # Informasi Dataset
    # ===============================

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Jumlah Missing Value", df.isnull().sum().sum())

    st.divider()

    # ===============================
    # Preview Dataset
    # ===============================

    st.subheader("📄 Preview Dataset")

    st.dataframe(df.head())

    st.divider()

    # ===============================
    # Statistik Dataset
    # ===============================

    st.subheader("📈 Statistik Dataset")

    st.dataframe(df.describe())

    st.divider()

    # ===============================
    # Missing Value
    # ===============================

    st.subheader("📌 Missing Value")

    missing = df.isnull().sum()

    st.dataframe(missing)

    st.divider()

    # ===============================
    # Distribusi Target
    # ===============================

    st.subheader("🎯 Distribusi Target")

    fig, ax = plt.subplots(figsize=(6,4))

    df["Target"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Target")

    ax.set_ylabel("Jumlah")

    st.pyplot(fig)

    st.divider()

    st.subheader("📈 Distribusi Umur Mahasiswa")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(df["Age at enrollment"], bins=15)

    ax.set_xlabel("Age")

    ax.set_ylabel("Frekuensi")

    st.pyplot(fig)

    st.divider()

    st.subheader("📈 Distribusi Admission Grade")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(df["Admission grade"], bins=20)

    ax.set_xlabel("Admission Grade")

    ax.set_ylabel("Frekuensi")

    st.pyplot(fig)

# ===========================================
# MODEL DEMO
# ===========================================

elif menu == "🤖 Model Demo":

    st.title("🤖 Prediksi Mahasiswa Dropout")

    st.write("""
Silakan masukkan data mahasiswa pada form di bawah ini,
kemudian klik tombol **Prediksi**.
""")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        Curricular2Approved = st.number_input(
            "Curricular units 2nd sem (approved)",
            min_value=0,
            max_value=30,
            value=10
        )

        Curricular2Grade = st.number_input(
            "Curricular units 2nd sem (grade)",
            min_value=0.0,
            max_value=20.0,
            value=12.0
        )

        Curricular1Approved = st.number_input(
            "Curricular units 1st sem (approved)",
            min_value=0,
            max_value=30,
            value=10
        )

        Curricular1Grade = st.number_input(
            "Curricular units 1st sem (grade)",
            min_value=0.0,
            max_value=20.0,
            value=12.0
        )

        Tuition = st.selectbox(
            "Tuition Fees Up To Date",
            [0,1],
            format_func=lambda x: "Lunas" if x==1 else "Belum Lunas"
        )

    with col2:

        Age = st.number_input(
            "Age at enrollment",
            min_value=15,
            max_value=70,
            value=20
        )

        Curricular2Evaluations = st.number_input(
            "Curricular units 2nd sem (evaluations)",
            min_value=0,
            max_value=30,
            value=10
        )

        AdmissionGrade = st.number_input(
            "Admission grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0
        )

        Course = st.number_input(
            "Course",
            min_value=1,
            max_value=20,
            value=1
        )

        Curricular1Evaluations = st.number_input(
            "Curricular units 1st sem (evaluations)",
            min_value=0,
            max_value=30,
            value=10
        )

    st.divider()

    if st.button("🔍 Prediksi", use_container_width=True):

        data = pd.DataFrame([[
            Curricular2Approved,
            Curricular2Grade,
            Curricular1Approved,
            Curricular1Grade,
            Tuition,
            Age,
            Curricular2Evaluations,
            AdmissionGrade,
            Course,
            Curricular1Evaluations
        ]], columns=[
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Tuition fees up to date",
            "Age at enrollment",
            "Curricular units 2nd sem (evaluations)",
            "Admission grade",
            "Course",
            "Curricular units 1st sem (evaluations)"
        ])

        data_scaled = scaler.transform(data)

        hasil = model.predict(data_scaled)

        probabilitas = model.predict_proba(data_scaled)[0]

        st.subheader("📊 Hasil Prediksi")

        if hasil[0] == 1:

            st.error("❌ Mahasiswa diprediksi **DROPOUT**")

        else:

            st.success("✅ Mahasiswa diprediksi **TIDAK DROPOUT**")

        st.write("### Probabilitas Prediksi")

        st.progress(float(max(probabilitas)))

        col1, col2 = st.columns(2)

        col1.metric(
            "Probabilitas Tidak Dropout",
            f"{probabilitas[0]*100:.2f}%"
        )

        col2.metric(
            "Probabilitas Dropout",
            f"{probabilitas[1]*100:.2f}%"
        )

        st.divider()

        st.subheader("📋 Data Input")

        st.dataframe(data)

# ===========================================
# EVALUASI MODEL
# ===========================================

elif menu == "📈 Evaluasi Model":

    st.title("📈 Evaluasi Model Random Forest")

    st.write("""
Halaman ini menampilkan hasil evaluasi model Random Forest yang digunakan
untuk memprediksi mahasiswa Dropout.
""")

    st.divider()

    col1, col2 = st.columns(2)

    col3, col4 = st.columns(2)

    col1.metric("Accuracy", "91.46 %")

    col2.metric("Precision", "90 %")

    col3.metric("Recall", "97 %")

    col4.metric("F1 Score", "93 %")

    st.divider()

    st.subheader("Kesimpulan")

    st.success("""
Model Random Forest memiliki performa yang sangat baik.

Accuracy sebesar 91.46% menunjukkan bahwa model mampu mengklasifikasikan
status mahasiswa dengan tingkat ketepatan yang tinggi.

Recall yang tinggi menunjukkan model sangat baik dalam mendeteksi mahasiswa
yang berpotensi mengalami Dropout.
""")
    
# ===========================================
# INTERPRETASI
# ===========================================

elif menu == "💡 Interpretasi Hasil":

    st.title("💡 Interpretasi Hasil")

    st.write("""
Model Random Forest menunjukkan bahwa beberapa faktor akademik memiliki
pengaruh besar terhadap kemungkinan mahasiswa mengalami Dropout.
""")

    st.markdown("""
### Faktor yang paling berpengaruh

- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)
- Curricular units 1st sem (approved)
- Tuition fees up to date
- Admission grade
- Age at enrollment

### Kesimpulan

Semakin banyak mata kuliah yang lulus pada semester pertama dan kedua,
serta nilai akademik yang tinggi, maka kemungkinan mahasiswa untuk
tidak mengalami Dropout semakin besar.

Sebaliknya, mahasiswa dengan nilai rendah, sedikit mata kuliah yang lulus,
dan tunggakan biaya kuliah memiliki risiko Dropout yang lebih tinggi.
""")
    

# ===========================================
# DOKUMENTASI
# ===========================================

elif menu == "📖 Dokumentasi":

    st.title("📖 Dokumentasi")

    st.markdown("""
## Judul

Prediksi Mahasiswa Dropout Menggunakan Algoritma Random Forest

---

## Dataset

Students Dropout and Academic Success Dataset

---

## Algoritma

Random Forest Classifier

---

## Bahasa Pemrograman

- Python
- Streamlit

---

## Library

- Pandas
- Scikit-Learn
- Matplotlib
- Joblib

---

##                          Developer

                             Dino Ageng Mintoro            (A11.2024.15925)

                             Sayyidah Syarifatul Ulya      (A11.2024.15590)

                                     Universitas Dian Nuswantoro
""")