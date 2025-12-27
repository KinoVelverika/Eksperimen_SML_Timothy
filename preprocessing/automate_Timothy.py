import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

#KONFIGURASI PATH
RAW_DATA_PATH = os.path.join('..', 'credit_risk_raw', 'credit_risk_dataset.csv')
OUTPUT_FILE = 'credit_risk_preprocessing.csv'

def load_data(path):
    print(f"📂 Memuat data mentah dari: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    print("⚙️ Sedang memproses data...")
    
    #Hapus Duplikat
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   -> Dihapus {initial_len - len(df)} data duplikat.")

    #Handling Missing Values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Isi numerik dengan Median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Isi kategorikal dengan Modus
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    #Encoding (Ubah Huruf ke Angka)
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    print("✅ Preprocessing selesai.")
    return df

if __name__ == "__main__":
    print("--- 🚀 MULAI OTOMASI DATA ---")
    
    try:
        # Cek apakah file raw ada
        if not os.path.exists(RAW_DATA_PATH):
            print(f"❌ Error: File tidak ditemukan di {RAW_DATA_PATH}")
            print("Pastikan nama folder 'credit_risk_raw' dan nama file CSV sudah benar.")
        else:
            # Eksekusi
            df = load_data(RAW_DATA_PATH)
            df_clean = preprocess_data(df)
            
            # Simpan Hasil
            df_clean.to_csv(OUTPUT_FILE, index=False)
            print(f"🎉 Sukses! Data bersih disimpan sebagai: {OUTPUT_FILE}")
            
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")