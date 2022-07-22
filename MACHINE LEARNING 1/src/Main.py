# Clustering Credit Card Dataset

# ==================== IMPORT LIBRARY ====================
from math import dist
import numpy as np # Mengolah data yang berbentuk array
import pandas as pd # Membaca file csv dan display tabel
import matplotlib.pyplot as plt
from scipy.sparse import data

# Membuat plot dalam bentuk diagram batang
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Silhouette score digunakan untuk mengukur kualitas cluster yang terbentuk
from sklearn.metrics import silhouette_score

# Standard Scaler digunakan untuk melakukan preprocessing
# menyeragamkan rentang nilai data pada dataset
from sklearn.preprocessing import StandardScaler

# Fungsi PCA dan Cosine_Similarity digunakan untuk reduksi fitur
# data sehingga dapat dilakukan visualisasi 2 dimensi
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ==================== IMPORT DATASET ====================
# from google.colab import drive
# drive.mount('/content/drive')
data_path = r'D:/LATIHAN PEMROGRAMAN/MACHINE LEARNING 1/CC GENERAL.csv'
dataset = pd.read_csv(data_path)

# Menghapus kolom cust_id
dataset.drop('CUST_ID', axis = 1, inplace = True)

# Pengecekan nilai null (kosong) pada dataset
# print(dataset.isnull().sum().sort_values(ascending=False).head())
dataset.isnull().sum().sort_values(ascending=False).head()


# ==================== PREPROCESSING DATA ====================
# 1. Mengisi nilai null pada dataset dengan nilai mean kolom tersebut
dataset.loc[(dataset['MINIMUM_PAYMENTS'].isnull()==True), 'MINIMUM_PAYMENTS']=dataset['MINIMUM_PAYMENTS'].mean()
dataset.loc[(dataset['CREDIT_LIMIT'].isnull()==True), 'CREDIT_LIMIT']=dataset['CREDIT_LIMIT'].mean()
dataset.isnull().sum().sort_values(ascending=False).head()

# 2. Menyeragamkan rentang nilai pada tiap-tiap fitur/kolom pada dataset
scaler = StandardScaler() # Memetakan nilai data yang baru berdasarkan perhitungan rata-rata dan variance dari tiap-tiap kolom
scaled_data = scaler.fit_transform(dataset)
df = pd.DataFrame(scaled_data, columns = dataset.columns)
df.head()


# ==================== VISUALISASI DATA ====================
# 1. Diagram dendogram
plt.figure(figsize=(15, 10))
dendrogram(linkage(df, method="ward"), leaf_rotation=90, p=5, color_threshold=20, leaf_font_size=10, truncate_mode='level')
plt.show()
""" Hierarki ini menghubungkan data-data yang 'jaraknya' dekat sebagai satu cluster 
hingga yang paling atas adalah garis yang menghubungkan dua buah cluster
yang tersisa.
Karena dataset terdiri dari 9000 record data, dilakukan pemangkasan data yang
ditampilkan dalam option 'truncate_mode'.
Apabila tidak dilakukan pemangkasan (truncate_mode = none), maka visualisasi
akan ditampilkan terhadap seluruh 9000 data. """

# 2. Silhouette score
"""
Untuk menentukan jumlah cluster terbaik, dilakukan perhitungan untuk jumlah cluster 2 - 10
Semakin tinggi nilai silhouette score, maka semakin baik cluster yang terbentuk
Hasil clustering dikatakan baik apabila data-data di dalam satu cluster berbeda jauh dengan
data-data cluster lainnya (memiliki inter-class variance yang tinggi). 
Serta apabila data-data didalam satu cluster mirip satu sama lain (memiliki intra-class variance yang rendah)
"""
silhouette_scores = []
for n_cluster in range(2, 10):
    silhouette_scores.append(
        silhouette_score(
            df, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(df)
        ) # Silhouette_score()
    ) # Silhouette_scores.append()


plt.bar(range(2, 10), silhouette_scores)
plt.xlabel('Jumlah cluster')
plt.ylabel('Silhouette Score')
plt.show()

# 3. Agglomerative clustering dengan jumlah cluster = 3
"""
Disini dilakukan clustering menggunakan metode Hierarchical Clustering
dengan jumlah cluster 3.
"""
agglo = AgglomerativeClustering(n_clusters=3)
agglo.fit(df)
labels = agglo.labels_
hasil_agglo = pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)
hasil_agglo.head()

# 4. Visualisasi hasil agglomerative clustering
""" Disini dilakukan visualisasi frekuensi nilai suatu fitur dari masing-masing cluster. """
for i in hasil_agglo:
    grid = sns.FacetGrid(hasil_agglo, col='cluster')
    grid.map(plt.hist, i)

# 5. Dekomposisi PCA
dist = 1 - cosine_similarity(df)
pca = PCA(n_components = 2)
pca = pca.fit_transform(dist)

# 6. Visualisasi penyebaran agglomerative clustering
x, y = pca[:, 0], pca[:, 1]
warna = { 0 : 'red', 1 : 'blue', 2 : 'green'}
label_pca = { 0 : 'cluster 0', 1 : 'cluster 1', 2 : 'cluster 2'}
df = pd.DataFrame({'x' : x, 'y' : y, 'label' : labels})
groups = df.groupby('label')
fig, ax = plt.subplots(figsize=(15, 10))

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, color = warna[name], label = label_pca[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

ax.legend()
ax.set_title("Visualisasi Agglomerative Clustering pada Dataset Kartu Kredit")
plt.show()
