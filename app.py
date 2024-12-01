import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Data Analysis App", 
    layout="wide", 
    page_icon="📊"
)

# CSS untuk mempercantik aplikasi
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding: 10px 30px;
    }
    .stButton>button {
        background-color: #4CAF50; 
        color: white; 
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Fungsi untuk menampilkan halaman login
def show_login_form():
    st.write("### 🔐 Welcome to Data Analysis App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "yoelsimbolon" and password == "foxyoel":
            st.session_state['logged_in'] = True
            st.success("Login berhasil!")
        else:
            st.error("Username atau password salah. Coba lagi.")

# Fungsi untuk memuat dataset
def load_dataset(upload_key):
    uploaded_file = st.file_uploader(f"Upload Dataset (CSV) untuk {upload_key}", type=["csv"], key=upload_key)
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil dimuat!")
            return data
        except Exception as e:
            st.error(f"Gagal memuat dataset: {e}")
            return None
    return None

# Fungsi untuk menangani missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy="mean")
    columns_with_na = data.columns[data.isnull().any()]
    
    if len(columns_with_na) > 0:
        st.warning(f"Kolom berikut memiliki missing values dan akan diimputasi: {columns_with_na}")
        data[columns_with_na] = imputer.fit_transform(data[columns_with_na])
    else:
        st.info("Tidak ada missing values dalam dataset.")
    return data

# Fungsi untuk menentukan apakah target variabel adalah klasifikasi atau regresi
def is_classification(target):
    return target.nunique() < 20

# Fungsi utama aplikasi
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_form()
else:
    # Navigasi Samping
    st.sidebar.title("🔗 Navigasi")
    option = st.sidebar.radio(
        "Pilih Proses:",
        ["📂 Data Preparation", "📊 EDA", "📈 Modeling", "🔮 Prediction"]
    )

    if option == "📂 Data Preparation":
        st.title("📂 Data Preparation")
        st.write("### Membersihkan Dataset")
        data = load_dataset("Data Preparation")
        
        if data is not None:
            st.write("### Dataset Overview")
            st.dataframe(data.head())
            
            # Menangani missing values
            data = handle_missing_values(data)
            st.write("### Dataset Setelah Imputasi")
            st.dataframe(data.head())
            
            if st.button("💾 Simpan Dataset Bersih"):
                data.to_csv("Cleaned_Dataset.csv", index=False)
                st.success("Dataset berhasil disimpan!")

    elif option == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")
        st.write("### Analisis Data")
        data = load_dataset("EDA")
        
        if data is not None:
            st.write("### Dataset Overview")
            st.dataframe(data.head())
            
            # Visualisasi distribusi data
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 0:
                selected_col = st.selectbox("Pilih Kolom untuk Visualisasi Distribusi:", numerical_cols)
                if selected_col:
                    st.write(f"Distribusi Kolom: {selected_col}")
                    fig = px.histogram(data, x=selected_col, nbins=30, title=f"Distribusi {selected_col}")
                    st.plotly_chart(fig)
                
                if st.checkbox("Tampilkan Heatmap Korelasi"):
                    fig_corr, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig_corr)
            else:
                st.warning("Dataset tidak memiliki kolom numerik untuk analisis.")
            
            if st.checkbox("Tampilkan Statistik Deskriptif"):
                st.write("Statistik Deskriptif")
                st.dataframe(data.describe())

    elif option == "📈 Modeling":
        st.title("📈 Modeling")
        st.write("### Latih Model")
        data = load_dataset("Modeling")
        
        if data is not None:
            st.write("### Dataset Overview")
            st.dataframe(data.head())
            
            target = st.selectbox("Pilih Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]
            
            if is_classification(y):
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                st.success("Model klasifikasi berhasil dilatih!")
                
                y_pred = model.predict(X)
                st.write("### Evaluasi Model")
                st.metric("Akurasi", f"{accuracy_score(y, y_pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y, y_pred))
            else:
                model = LinearRegression()
                model.fit(X, y)
                st.success("Model regresi berhasil dilatih!")
                
                y_pred = model.predict(X)
                st.write("### Evaluasi Model")
                st.metric("R² Score", f"{r2_score(y, y_pred):.2f}")
                st.write("### Mean Squared Error")
                st.metric("MSE", f"{mean_squared_error(y, y_pred):.2f}")

    elif option == "🔮 Prediction":
        st.title("🔮 Prediction")
        st.write("### Prediksi dengan Dataset Baru")
        train_data = load_dataset("Prediction")
        
        if train_data is not None:
            st.write("### Dataset Overview")
            st.dataframe(train_data.head())
            
            target = st.selectbox("Pilih Target Variable (Pelatihan):", train_data.columns)
            features = train_data.drop(columns=[target])
            X_train = features.select_dtypes(include=['float64', 'int64'])
            y_train = train_data[target]
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            st.success("Model berhasil dilatih!")
            
            # Input data untuk prediksi
            st.write("Masukkan data untuk prediksi.")
            pred_data = {}
            for col in X_train.columns:
                pred_data[col] = st.number_input(f"Nilai untuk {col}", value=0.0)
            
            if st.button("Prediksi"):
                input_data = pd.DataFrame([pred_data])
                prediction = model.predict(input_data)
                st.write(f"Prediksi untuk data ini: {prediction[0]}")
    



