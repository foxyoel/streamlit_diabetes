import pandas as pd

# Ganti path dengan path dataset di komputer Anda
dataset_path = 'C:/Streamlit/streamlit_project/diabetes_prediction_dataset.csv'

# Load the dataset
data = pd.read_csv(dataset_path)  # Memuat dataset

# Cek kolom yang ada dalam dataset
print("Kolom dalam dataset:", data.columns)

# Menghapus spasi atau karakter tak terlihat pada nama kolom
data.columns = data.columns.str.strip()

# Periksa apakah kolom 'diabetes' ada
if 'diabetes' in data.columns:
    print("'diabetes' kolom ditemukan.")
else:
    print("'diabetes' kolom TIDAK ditemukan.")

# Jika kolom 'diabetes' ditemukan, lanjutkan pemrosesan
if 'diabetes' in data.columns:
    # Pisahkan fitur dan target variabel
    X = data.drop(columns=['diabetes'])  # 'diabetes' sebagai kolom target
    y = data['diabetes']  # Target variabel
    
    # Lakukan operasi lebih lanjut seperti training model di sini
    print("Dataset berhasil diproses.")
else:
    print("Kolom 'diabetes' tidak ditemukan, silakan periksa dataset Anda.")
