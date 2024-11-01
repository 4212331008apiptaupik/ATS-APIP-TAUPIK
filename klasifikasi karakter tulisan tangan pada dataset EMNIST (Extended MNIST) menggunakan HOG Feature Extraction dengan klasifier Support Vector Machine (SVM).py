import pandas as pd

# Gantilah 'nama_file.csv' dengan nama file CSV Anda
data = pd.read_csv('emnist-bymerge-test.csv')

# Menampilkan data untuk memverifikasi bahwa CSV telah berhasil dibaca
print(data)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Baca file CSV
data = pd.read_csv('emnist-bymerge-test.csv')

# Pilih gambar pertama untuk ditampilkan (misalnya baris pertama)
# Asumsi: Kolom pertama adalah label dan sisanya adalah piksel
label = data.iloc[0, 0]            # Label untuk gambar
pixels = data.iloc[0, 1:].values    # Data piksel

# Ubah data piksel menjadi matriks 2D (28x28)
image = pixels.reshape(28, 28)

# Tampilkan gambar
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')  # Hilangkan sumbu
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Membaca file CSV
image_data = pd.read_csv('emnist-bymerge-test.csv')

# Misalkan kolom pertama adalah label dan kolom sisanya adalah piksel
# Misalkan setiap gambar adalah 28x28 (misalnya, jika dataset EMNIST)
images = image_data.iloc[:, 1:].values.reshape(-1, 28, 28)  # Mengubah ke format (n, 28, 28)
labels = image_data.iloc[:, 0].values  # Kolom pertama sebagai label

# Ambil satu gambar untuk contoh
sample_image = images[0]  # Ambil gambar pertama
sample_label = labels[0]   # Ambil label untuk gambar pertama

# Ekstraksi fitur HOG
hog_features, hog_image = hog(sample_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Meningkatkan kontras dari gambar HOG
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Menampilkan gambar asli dan gambar HOG
plt.figure(figsize=(10, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.imshow(sample_image, cmap='gray')
plt.title(f'Gambar Asli - Label: {sample_label}')
plt.axis('off')

# Gambar HOG
plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('Gambar HOG')
plt.axis('off')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset dari CSV (misalnya: 'emnist-bymerge-test.csv')
data = pd.read_csv('emnist-bymerge-test.csv')

# Data EMNIST terdiri dari label di kolom pertama dan piksel di kolom berikutnya
y = data.iloc[:, 0].values           # Label
X = data.iloc[:, 1:].values           # Gambar dalam bentuk 28x28 piksel yang diratakan

# Memilih salah satu gambar untuk ditampilkan dan melakukan ekstraksi HOG
sample_image = X[0].reshape(28, 28)   # Ambil gambar pertama dan ubah menjadi bentuk 28x28

# Ekstraksi fitur HOG dan visualisasi
fd, hog_image = hog(sample_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Menampilkan gambar asli dan HOG-nya
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax1.imshow(sample_image, cmap=plt.cm.gray)
ax1.set_title('Original Image')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')
plt.show()

# Ekstraksi Fitur HOG untuk semua gambar
def extract_hog_features(images):
    hog_features = []
    for image in images:
        image = image.reshape(28, 28)  # Pastikan setiap gambar berbentuk 28x28
        feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(feature)
    return np.array(hog_features)

# Ekstraksi fitur HOG untuk data latih
X_hog = extract_hog_features(X)

# Membuat model SVM dengan pipeline (normalisasi + SVM)
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Membagi data latih dan data uji (misalnya: 80% data latih, 20% data uji)
split_index = int(0.8 * len(X_hog))
X_train, X_test = X_hog[:split_index], X_hog[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Melatih model
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Menampilkan laporan hasil klasifikasi
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
