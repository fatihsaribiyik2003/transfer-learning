import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. AYARLAR ---
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'

# Resimler 128x128 piksel boyutuna getirilecek
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("\n🚀 Veriler yükleniyor...")

# --- 2. VERİ YÜKLEME (DATA PIPELINE) ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# --- 3. MODELİ OLUŞTURMA (TRANSFER LEARNING) ---
print("🧠 VGG16 Modeli hazırlanıyor...")

base_model = VGG16(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- 4. MODELİ DERLEME ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# --- 5. EĞİTİMİ BAŞLAT ---
print("\n🏋️ Eğitim başlıyor! (Mac performansına göre biraz sürebilir)...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)

# --- 6. SONUÇLARI GÖRSELLEŞTİR VE KAYDET ---
print("\n📊 Grafikler hazırlanıyor...")

history_frame = pd.DataFrame(history.history)

# Grafik çerçevesini oluştur
plt.figure(figsize=(12, 6))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history_frame['loss'], label='Eğitim Kaybı')
plt.plot(history_frame['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp (Loss) Grafiği')
plt.xlabel('Epoch (Tur)')
plt.legend()

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history_frame['binary_accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_frame['val_binary_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk (Accuracy) Grafiği')
plt.xlabel('Epoch (Tur)')
plt.legend()

# --- BURASI DEĞİŞTİ ---
# 1. Önce Kaydet (show'dan önce olmalı!)
output_path = 'egitim_sonuclari.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"💾 Grafik '{output_path}' olarak kaydedildi.")

# 2. Sonra Ekranda Göster
plt.show()

print("✅ İşlem Tamamlandı!")