print("Script is running...")

import pandas as pd

# ----------------------------MEMBACA DATASET-------------------------
test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")
print(test.head())
print(train.head())

# Menampilkan ringkasan informasi dari dataset
print("\nMenampilkan ringkasan informasi dari dataset:")
train.info()

# Menampilkan statistik deskriptif dari dataset
print("\nMenampilkan statistik deskriptif dari dataset:")
print(train.describe(include="all"))

# Memeriksa jumlah nilai yang hilang di setiap kolom
print("\nMemeriksa jumlah nilai yang hilang di setiap kolom:")
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])

# Memisahkan kolomm yang memiliki missing value lebih dari 75% dan kurang dari 75%
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

# Contoh mengisi nilai yang hilang dengan median untuk kolom numerik
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

# Contoh mengisi nilai yang hilang dengan median untuk kolom kategori
kategorical_features = train[less].select_dtypes(include=['object']).columns
 
for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

# Menghapus kolom dengan terlalu banyak nilai yang hilang
df = train.drop(columns=over)

# Verifikasi jumlah nilai yang hilang setelah dihapus
print("\ncek ulang missing value setelah cleaning data:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


#-------------------------MENANGANI OUTLIERS-------------------------
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    # plt.show()

# Contoh sederhana untuk mengidentifikasi outliers menggunakan IQR
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

#hapus outlier berdasarkan perhitungan di atas.

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_features]
 
# Menggabungkan kembali dengan kolom kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)


# Pro Tips

# Jika Anda tidak ingin menghapus outliers seperti contoh di atas, silakan gunakan metode agregasi seperti berikut.

# ```
# median = df['column_name'].median()
# df['column_name'] = df['column_name'].apply(lambda x: median if x < (Q1 - 1.5  IQR) or x > (Q3 + 1.5  IQR) else x)
# ```
# atau

# ```
# # Mengganti outlier dengan nilai batas terdekat
# df['column_name'] = df['column_name'].apply(lambda x: (Q1 - 1.5  IQR) if x < lower_bound else (Q3 + 1.5  IQR) if x > (Q3 + 1.5 * IQR) else x)
 
 
# ```


#---------- LAKUKAN STANDARDISASI/NORMALISASI PADA KOLOM NUMERIK JIKA ADA ----------------
from sklearn.preprocessing import StandardScaler
 
# Standardisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Histogram Sebelum Standardisasi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train[numeric_features[3]], kde=True)
plt.title("Histogram Sebelum Standardisasi")

# Histogram Setelah Standardisasi
plt.subplot(1, 2, 2)
sns.histplot(df[numeric_features[3]], kde=True)
plt.title("Histogram Setelah Standardisasi")

#-------------------------MENANGANI DUPLIKASI DATA-------------------------
# Mengidentifikasi baris duplikat
duplicates = df.duplicated()
 
print("\nBaris duplikat:")
print(df[duplicates])

# Menghapus baris duplikat
df = df.drop_duplicates()
 
print("\nDataFrame setelah menghapus duplikat:")
print(df)

#-------------------------KOVERSI TIPE DATA KATEGORI (ENCODING)-------------------------
print("\nTipe data kategori sebelum encoding:")
category_features = df.select_dtypes(include=['object']).columns
print(df[category_features])

# One-hot encoding
df_one_hot = pd.get_dummies(df, columns=category_features)
print("\none hot encoding: ")
print(df_one_hot)

# Label encoding
from sklearn.preprocessing import LabelEncoder
 
# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)
 
for col in category_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])
 
# Menampilkan hasil
print("\nlabel encoder:")
print(df_lencoder)
